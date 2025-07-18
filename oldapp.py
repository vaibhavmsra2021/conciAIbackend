import os
import json
import asyncio
import logging
import io
import base64
from datetime import datetime
from typing import Optional, Dict, Any
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from supabase.client import create_client, Client
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import re
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub import AudioSegment

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ConciAI Voice Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize Supabase client (only if credentials are provided)
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None
else:
    logger.warning("Supabase credentials not provided. Database features will be disabled.")

# Initialize Gemini AI (only if API key is provided)
model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Gemini AI initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini AI: {e}")
        model = None
else:
    logger.warning("Gemini API key not provided. AI responses will be disabled.")

# Initialize speech components
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def convert_audio_to_wav(audio_bytes: bytes) -> bytes:
    """Convert audio bytes to WAV format for speech recognition"""
    try:
        logger.info(f"Converting audio: {len(audio_bytes)} bytes")
        logger.info(f"Audio header: {audio_bytes[:20].hex()}")
        
        # Try to load the audio with pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        logger.info(f"Audio loaded: {len(audio)}ms, {audio.channels} channels, {audio.frame_rate}Hz")
        
        # Export as WAV with specific parameters for speech recognition
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav', parameters=[
            '-ac', '1',  # Mono
            '-ar', '16000',  # 16kHz sample rate
            '-sample_fmt', 's16'  # 16-bit PCM
        ])
        wav_io.seek(0)
        wav_bytes = wav_io.read()
        
        logger.info(f"Converted to WAV: {len(wav_bytes)} bytes")
        return wav_bytes
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        # Return original bytes as fallback
        return audio_bytes

# Hotel context prompt
HOTEL_CONTEXT = """
You are ConciAI, a friendly hotel voice assistant. You help guests with hotel services and information.

IMPORTANT INSTRUCTIONS:
1. Keep all responses SHORT and CRISP (maximum 2-3 sentences)
2. Be warm and professional
3. For GENERAL QUERIES (restaurant hours, weather, local info), provide direct helpful answers
4. For SERVICE REQUESTS (towels, maintenance, room service), respond helpfully and staff will be notified automatically
5. For COMPLAINTS or URGENT issues, respond empathetically and staff will be notified immediately

HOTEL SERVICES AVAILABLE:
- Room service (24/7)
- Housekeeping
- Concierge services
- Restaurant reservations
- Spa appointments
- Laundry service
- Wake-up calls
- Transportation
- Local attractions information
- Weather information

SAMPLE RESPONSES:
- "The restaurant is open until 10 PM. Would you like me to make a reservation?"
- "I'll arrange room service for you right away. Is there anything specific you'd like to order?"
- "I'll notify housekeeping about the towel request. They'll be with you within 15 minutes."
- "The weather today is sunny with a high of 75°F. Perfect for outdoor activities!"

Remember: Be concise, helpful, and always maintain a warm, professional tone.
"""

class VoiceAssistant:
    def __init__(self):
        self.is_listening = False
        self.wake_word = "hey conci"
        
    async def get_user_by_room(self, room_number: str) -> Optional[Dict]:
        """Fetch user details by room number"""
        if not supabase:
            logger.warning("Supabase not configured. Cannot fetch user data.")
            return None
        try:
            response = supabase.table('users').select('*').eq('room_number', room_number).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            return None
    
    async def detect_request_or_complaint(self, text: str) -> bool:
        """Detect if the text contains a service request or complaint that needs staff attention"""
        text_lower = text.lower()
        
        # Action words that indicate a request for service
        action_words = [
            'need', 'want', 'request', 'send', 'bring', 'get', 'call',
            'fix', 'repair', 'replace', 'clean', 'change', 'check',
            'urgent', 'emergency', 'asap', 'immediately', 'now', 'right away'
        ]
        
        # Specific service items that require staff action
        service_items = [
            'towels', 'room service', 'housekeeping', 'maintenance',
            'laundry', 'water', 'electricity', 'ac', 'heating', 'plumbing',
            'toilet', 'shower', 'sink', 'light', 'tv', 'wifi',
            'key', 'card', 'lock', 'door', 'window', 'curtain',
            'pillow', 'blanket', 'sheet', 'soap', 'shampoo',
            'toilet paper', 'tissue', 'trash', 'garbage'
        ]
        
        # Complaint indicators
        complaint_words = [
            'complaint', 'problem', 'issue', 'broken', 'not working',
            'disappointed', 'unhappy', 'terrible', 'awful', 'wrong',
            'bad', 'poor', 'unacceptable', 'noise', 'smell', 'dirty',
            'mess', 'uncomfortable'
        ]
        
        # Check for urgent language (always creates a request)
        urgent_words = ['urgent', 'emergency', 'asap', 'immediately', 'now', 'right away']
        is_urgent = any(word in text_lower for word in urgent_words)
        
        # Check for complaints (always creates a request)
        has_complaint = any(word in text_lower for word in complaint_words)
        
        # Check for service requests (must have both action word AND service item)
        has_action_word = any(word in text_lower for word in action_words)
        has_service_item = any(item in text_lower for item in service_items)
        has_service_request = has_action_word and has_service_item
        
        # Log the detection for debugging
        logger.info(f"Text: '{text}'")
        logger.info(f"Has action word: {has_action_word}")
        logger.info(f"Has service item: {has_service_item}")
        logger.info(f"Has complaint: {has_complaint}")
        logger.info(f"Is urgent: {is_urgent}")
        logger.info(f"Will create request: {has_service_request or has_complaint or is_urgent}")
        
        return has_service_request or has_complaint or is_urgent
    
    async def classify_request_type(self, text: str) -> str:
        """Classify if it's a request or complaint"""
        complaint_keywords = [
            'complaint', 'problem', 'issue', 'broken', 'not working',
            'disappointed', 'unhappy', 'terrible', 'awful', 'wrong',
            'bad', 'poor', 'unacceptable'
        ]
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in complaint_keywords):
            return 'complaint'
        return 'request'
    
    async def determine_priority(self, text: str) -> str:
        """Determine priority level of the request"""
        high_priority = ['urgent', 'emergency', 'immediately', 'asap', 'broken', 'not working']
        medium_priority = ['soon', 'today', 'problem', 'issue']
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in high_priority):
            return 'high'
        elif any(keyword in text_lower for keyword in medium_priority):
            return 'medium'
        return 'low'
    
    async def save_request_to_db(self, user_id: str, message: str, request_type: str, priority: str):
        """Save request/complaint to database"""
        if not supabase:
            logger.warning("Supabase not configured. Cannot save request to database.")
            return False
        try:
            data = {
                'user_id': user_id,
                'type': request_type,
                'message': message,
                'status': 'pending',
                'priority': priority,
                'created_at': datetime.utcnow().isoformat()
            }
            
            response = supabase.table('requests').insert(data).execute()
            logger.info(f"Request saved to database: {response.data}")
            return True
        except Exception as e:
            logger.error(f"Error saving request: {e}")
            return False
    
    async def generate_ai_response(self, text: str, user_data: Optional[Dict] = None) -> str:
        """Generate AI response using Gemini"""
        if not model:
            logger.error("Gemini model not initialized")
            return "I apologize, but AI services are currently unavailable. Please contact the front desk for assistance."
        try:
            # Personalize the prompt if user data is available
            personalized_context = HOTEL_CONTEXT
            if user_data:
                personalized_context += f"\n\nCURRENT GUEST: {user_data['name']} in Room {user_data['room_number']}"
            
            prompt = f"{personalized_context}\n\nGuest message: {text}\n\nYour response:"
            logger.info(f"Generating AI response for: {text}")
            logger.info(f"Using model: {model}")
            
            # Generate response with safety settings
            response = model.generate_content(
                prompt,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            )
            
            if response and hasattr(response, 'text'):
                ai_response = response.text.strip()
                logger.info(f"AI Response: {ai_response}")
                return ai_response
            else:
                logger.error(f"Invalid response from Gemini: {response}")
                return "I apologize, but I'm having technical difficulties. Please contact the front desk for assistance."
                
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "I apologize, but I'm having technical difficulties. Please contact the front desk for assistance."
    
    async def speak_text(self, text: str):
        """Convert text to speech"""
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
    
    async def speak_text_to_bytes(self, text: str) -> bytes:
        """Convert text to speech and return audio bytes (WAV format)"""
        import tempfile
        import os
        
        # Use a temporary file to save the TTS output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
            temp_path = tf.name
        try:
            tts_engine.save_to_file(text, temp_path)
            tts_engine.runAndWait()
            # Read the WAV file into bytes
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            logger.info(f"Generated TTS audio: {len(audio_bytes)} bytes")
            return audio_bytes
        except Exception as e:
            logger.error(f"Error generating TTS: {e}")
            return b''  # Return empty bytes if TTS fails
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def process_voice_input_text(self, text_input: str, room_number: str) -> Dict[str, Any]:
        """Process text input and return response, with TTS audio"""
        try:
            user_data = await self.get_user_by_room(room_number)
            is_request = await self.detect_request_or_complaint(text_input)
            ai_response = await self.generate_ai_response(text_input, user_data)
            if is_request and user_data:
                request_type = await self.classify_request_type(text_input)
                priority = await self.determine_priority(text_input)
                await self.save_request_to_db(
                    user_data['id'], text_input, request_type, priority
                )
            # Generate TTS audio
            audio_bytes = await self.speak_text_to_bytes(ai_response)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return {
                "recognized_text": text_input,
                "ai_response": ai_response,
                "is_request": is_request,
                "user_data": user_data,
                "audio_base64": audio_base64,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            return {"error": str(e), "status": "error"}

    async def process_voice_input_audio(self, audio_data: str, room_number: str) -> Dict[str, Any]:
        """Process audio input and return response, with TTS audio"""
        try:
            audio_bytes = base64.b64decode(audio_data)
            logger.info(f"Received audio data: {len(audio_bytes)} bytes")
            logger.info(f"Audio bytes header: {audio_bytes[:20]}")
            
            # Check if audio data is valid
            if len(audio_bytes) < 100:  # Too small to be valid audio
                logger.warning("Audio data too small, using fallback")
                recognized_text = "audio data too small"
            else:
                # Convert audio to WAV format for better speech recognition
                wav_bytes = convert_audio_to_wav(audio_bytes)
                recognized_text = await self.recognize_speech(wav_bytes)
            
            if not recognized_text or recognized_text in ["speech recognition failed", "speech recognition service error", "speech recognition error", "audio data too small"]:
                logger.warning(f"Speech recognition failed: '{recognized_text}'")
                return {
                    "error": f"Could not recognize speech: {recognized_text}",
                    "status": "error"
                }
            user_data = await self.get_user_by_room(room_number)
            is_request = await self.detect_request_or_complaint(recognized_text)
            ai_response = await self.generate_ai_response(recognized_text, user_data)
            if is_request and user_data:
                request_type = await self.classify_request_type(recognized_text)
                priority = await self.determine_priority(recognized_text)
                await self.save_request_to_db(
                    user_data['id'], recognized_text, request_type, priority
                )
            # Generate TTS audio
            audio_bytes = await self.speak_text_to_bytes(ai_response)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return {
                "recognized_text": recognized_text,
                "ai_response": ai_response,
                "is_request": is_request,
                "user_data": user_data,
                "audio_base64": audio_base64,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
            return {"error": str(e), "status": "error"}

    async def recognize_speech(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to text using speech recognition"""
        try:
            # Create an audio file object from bytes
            audio_file = io.BytesIO(audio_bytes)
            
            # Use speech recognition to convert audio to text
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise and set timeout
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.record(source)
                
                logger.info(f"Attempting speech recognition on {len(audio_bytes)} bytes of audio")
                
                # Try multiple recognition methods
                try:
                    # First try Google Speech Recognition
                    text = recognizer.recognize_google(audio, language='en-US')
                    logger.info(f"Speech recognized successfully: '{text}'")
                    return str(text)
                except sr.UnknownValueError:
                    logger.warning("Google Speech Recognition could not understand audio")
                    # Try with different language or settings
                    try:
                        text = recognizer.recognize_google(audio, language='en-US', show_all=True)
                        if text and len(text) > 0:
                            logger.info(f"Speech recognized (alternative): {text}")
                            return str(text[0]['transcript'])
                    except Exception as e:
                        logger.error(f"Alternative recognition failed: {e}")
                    
                    # If still no result, return a more descriptive placeholder
                    logger.warning("Speech recognition failed - using fallback")
                    return "speech recognition failed"
                    
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return "speech recognition service error"
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "speech recognition error"

    async def process_voice_input(self, audio_data: bytes, room_number: str) -> Dict[str, Any]:
        """Process voice input and return response (legacy method)"""
        try:
            # Convert audio to text (simplified - in real implementation, you'd process the audio_data)
            # For now, we'll simulate this with a placeholder
            recognized_text = "This is a placeholder for voice recognition"
            
            # Get user data
            user_data = await self.get_user_by_room(room_number)
            
            # Check if it's a request or complaint
            is_request = await self.detect_request_or_complaint(recognized_text)
            
            # Generate AI response
            ai_response = await self.generate_ai_response(recognized_text, user_data)
            
            # If it's a request/complaint, save to database
            if is_request and user_data:
                request_type = await self.classify_request_type(recognized_text)
                priority = await self.determine_priority(recognized_text)
                await self.save_request_to_db(
                    user_data['id'], 
                    recognized_text, 
                    request_type, 
                    priority
                )
            
            return {
                "recognized_text": recognized_text,
                "ai_response": ai_response,
                "is_request": is_request,
                "user_data": user_data,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            return {
                "error": str(e),
                "status": "error"
            }

# Initialize voice assistant
voice_assistant = VoiceAssistant()

def convert_to_wav(audio_bytes):
    # Try to load as OGG/WEBM, fallback to WAV
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav')
        wav_io.seek(0)
        return wav_io.read()
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return audio_bytes  # fallback

@app.get("/")
async def root():
    return {"message": "ConciAI Voice Assistant is running"}

@app.get("/test")
async def test_client():
    """Serve the test client HTML page"""
    return FileResponse("test_client.html")

@app.get("/realtime")
async def realtime_client():
    """Serve the real-time voice client HTML page"""
    return FileResponse("realtime_client.html")

@app.get("/simple")
async def simple_client():
    """Serve the simple voice client HTML page"""
    return FileResponse("simple_client.html")

@app.get("/webspeech")
async def webspeech_client():
    """Serve the Web Speech API client HTML page"""
    return FileResponse("webspeech_client.html")

@app.get("/conversation")
async def conversation_client():
    """Serve the conversation client HTML page"""
    return FileResponse("conversation_client.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/voice/process")
async def process_voice(room_number: str, text_input: Optional[str] = None):
    """Process voice input from ESP32 or text input for testing"""
    try:
        if text_input:
            # For testing with text input
            user_data = await voice_assistant.get_user_by_room(room_number)
            is_request = await voice_assistant.detect_request_or_complaint(text_input)
            ai_response = await voice_assistant.generate_ai_response(text_input, user_data)
            
            if is_request and user_data:
                request_type = await voice_assistant.classify_request_type(text_input)
                priority = await voice_assistant.determine_priority(text_input)
                await voice_assistant.save_request_to_db(
                    user_data['id'], 
                    text_input, 
                    request_type, 
                    priority
                )
            
            return {
                "recognized_text": text_input,
                "ai_response": ai_response,
                "is_request": is_request,
                "user_data": user_data,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=400, detail="No input provided")
            
    except Exception as e:
        logger.error(f"Error in voice processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{room_number}")
async def get_user_by_room(room_number: str):
    """Get user details by room number"""
    user_data = await voice_assistant.get_user_by_room(room_number)
    if user_data:
        return user_data
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.get("/test-ai")
async def test_ai():
    """Test AI response generation"""
    try:
        test_text = "Hello, I need fresh towels"
        response = await voice_assistant.generate_ai_response(test_text)
        return {
            "input": test_text,
            "ai_response": response,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error testing AI: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """WebSocket endpoint for real-time voice communication"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_json()
            
            # Extract data from message
            room_number = message.get("room_number", "101")  # Default room
            audio_data = message.get("audio_data")  # Base64 encoded audio
            text_input = message.get("text_input")  # Optional text input for testing
            
            try:
                if text_input:
                    # Process text input directly
                    result = await voice_assistant.process_voice_input_text(text_input, room_number)
                elif audio_data:
                    # Process audio data
                    result = await voice_assistant.process_voice_input_audio(audio_data, room_number)
                else:
                    result = {
                        "error": "No audio_data or text_input provided",
                        "status": "error"
                    }
                
                # Send response back to client
                await websocket.send_json(result)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                error_response = {
                    "error": str(e),
                    "status": "error"
                }
                await websocket.send_json(error_response)
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)