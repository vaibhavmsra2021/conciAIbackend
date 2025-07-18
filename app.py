import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
from dotenv import load_dotenv
load_dotenv()

# Add missing imports from oldapp.py
import io
import base64
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydub import AudioSegment

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

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize speech components
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Hotel context prompt
HOTEL_CONTEXT = """
You are ConciAI, a friendly hotel voice assistant. You help guests with hotel services and information.

IMPORTANT INSTRUCTIONS:
1. Keep all responses SHORT and CRISP (maximum 2-3 sentences)
2. Be warm and professional
3. If you detect a REQUEST or COMPLAINT, respond helpfully but mark it for staff attention
4. For general queries, provide direct helpful answers about hotel services

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
- "I'll arrange room service for you right away. Is there anything specific you'd like to order?"
- "I'll notify housekeeping about the towel request. They'll be with you within 15 minutes."
- "The restaurant is open until 10 PM. Would you like me to make a reservation?"

Remember: Be concise, helpful, and always maintain a warm, professional tone.
"""

# Add or update utility functions from oldapp.py

def convert_audio_to_wav(audio_bytes: bytes) -> bytes:
    """Convert audio bytes to WAV format for speech recognition"""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav', parameters=[
            '-ac', '1',  # Mono
            '-ar', '16000',  # 16kHz sample rate
            '-sample_fmt', 's16'  # 16-bit PCM
        ])
        wav_io.seek(0)
        return wav_io.read()
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return audio_bytes

# Add VoiceAssistant class from oldapp.py
class VoiceAssistant:
    def __init__(self):
        self.is_listening = False
        self.wake_word = "hey conci"
        
    async def get_user_by_room(self, room_number: str) -> Optional[Dict]:
        try:
            response = supabase.table('users').select('*').eq('room_number', room_number).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            return None
    
    async def detect_request_or_complaint(self, text: str) -> bool:
        text_lower = text.lower()
        action_words = [
            'need', 'want', 'request', 'send', 'bring', 'get', 'call',
            'order', 'book', 'reserve', 'arrange', 'deliver', 'provide', 'serve',
            'fix', 'repair', 'replace', 'clean', 'change', 'check',
            'urgent', 'emergency', 'asap', 'immediately', 'now', 'right away'
        ]
        service_items = [
            'towels', 'room service', 'housekeeping', 'maintenance',
            'laundry', 'water', 'water bottle', 'bottle', 'food', 'order', 'snack', 'drink', 'tea', 'coffee', 'breakfast', 'lunch', 'dinner',
            'restaurant', 'plumber', 'plumbing', 'electrician', 'ac', 'air conditioner', 'heating', 'spa', 'reservation',
            'toilet', 'shower', 'sink', 'light', 'tv', 'wifi',
            'key', 'card', 'lock', 'door', 'window', 'curtain',
            'pillow', 'blanket', 'sheet', 'soap', 'shampoo',
            'toilet paper', 'tissue', 'trash', 'garbage'
        ]
        complaint_words = [
            'complaint', 'problem', 'issue', 'broken', 'not working',
            'disappointed', 'unhappy', 'terrible', 'awful', 'wrong',
            'bad', 'poor', 'unacceptable', 'noise', 'smell', 'dirty',
            'mess', 'uncomfortable'
        ]
        urgent_words = ['urgent', 'emergency', 'asap', 'immediately', 'now', 'right away']
        is_urgent = any(word in text_lower for word in urgent_words)
        has_complaint = any(word in text_lower for word in complaint_words)
        has_action_word = any(word in text_lower for word in action_words)
        has_service_item = any(item in text_lower for item in service_items)
        has_service_request = has_action_word and has_service_item
        return has_service_request or has_complaint or is_urgent
    
    async def classify_request_type(self, text: str) -> str:
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
        high_priority = ['urgent', 'emergency', 'immediately', 'asap', 'broken', 'not working']
        medium_priority = ['soon', 'today', 'problem', 'issue']
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in high_priority):
            return 'high'
        elif any(keyword in text_lower for keyword in medium_priority):
            return 'medium'
        return 'low'
    
    async def save_request_to_db(self, user_id: str, message: str, request_type: str, priority: str, category_id: str = "", assigned_to: str = "", assigned_at: str = ""):
        try:
            data = {
                'user_id': user_id,
                'type': request_type,
                'message': message,
                'status': 'pending',
                'priority': priority,
                'created_at': datetime.utcnow().isoformat(),
                'category_id': category_id,
                'assigned_to': assigned_to,
                'assigned_at': assigned_at
            }
            response = supabase.table('requests').insert(data).execute()
            logger.info(f"Request saved to database: {response.data}")
            return True
        except Exception as e:
            logger.error(f"Error saving request: {e}")
            return False
    
    async def generate_ai_response(self, text: str, user_data: Optional[Dict] = None) -> str:
        try:
            personalized_context = HOTEL_CONTEXT
            if user_data:
                personalized_context += f"\n\nCURRENT GUEST: {user_data['name']} in Room {user_data['room_number']}"
            prompt = f"{personalized_context}\n\nGuest message: {text}\n\nYour response:"
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I apologize, but I'm having technical difficulties. Please contact the front desk for assistance."
    
    async def speak_text(self, text: str):
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
    
    async def speak_text_to_bytes(self, text: str) -> bytes:
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
            temp_path = tf.name
        try:
            tts_engine.save_to_file(text, temp_path)
            tts_engine.runAndWait()
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            return audio_bytes
        except Exception as e:
            logger.error(f"Error generating TTS: {e}")
            return b''
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def process_voice_input_text(self, text_input: str, room_number: str) -> Dict[str, Any]:
        try:
            user_data = await self.get_user_by_room(room_number)
            is_request = await self.detect_request_or_complaint(text_input)
            ai_response = await self.generate_ai_response(text_input, user_data)
            if is_request and user_data:
                request_type = await self.classify_request_type(text_input)
                priority = await self.determine_priority(text_input)
                # Determine category and auto-assign staff
                category_id = None
                assigned_to = None
                assigned_at = None
                # Try to get category from request_categories table
                try:
                    response = supabase.table('request_categories').select('*').execute()
                    categories = response.data if response.data else []
                    text_lower = text_input.lower()
                    for category in categories:
                        keywords = category.get('keywords', [])
                        if any(keyword.lower() in text_lower for keyword in keywords):
                            category_id = category['id']
                            break
                    # Auto-assign staff if category found
                    if category_id:
                        cat_response = supabase.table('request_categories').select('*').eq('id', category_id).single().execute()
                        if cat_response.data:
                            required_role = cat_response.data['assigned_role']
                            staff_response = supabase.table('staff_users').select('*').eq('role', required_role).eq('is_active', True).execute()
                            if staff_response.data:
                                assigned_to = staff_response.data[0]['id']
                                assigned_at = datetime.utcnow().isoformat()
                except Exception as e:
                    logger.error(f"Error in category/assignment: {e}")
                # Ensure all fields are strings for save_request_to_db
                category_id = category_id or ""
                assigned_to = assigned_to or ""
                assigned_at = assigned_at or ""
                await self.save_request_to_db(
                    user_data['id'], text_input, request_type, priority, category_id, assigned_to, assigned_at
                )
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
        try:
            audio_bytes = base64.b64decode(audio_data)
            if len(audio_bytes) < 100:
                recognized_text = "audio data too small"
            else:
                wav_bytes = convert_audio_to_wav(audio_bytes)
                recognized_text = "speech recognition not implemented"
            user_data = await self.get_user_by_room(room_number)
            is_request = await self.detect_request_or_complaint(recognized_text)
            ai_response = await self.generate_ai_response(recognized_text, user_data)
            if is_request and user_data:
                request_type = await self.classify_request_type(recognized_text)
                priority = await self.determine_priority(recognized_text)
                # Determine category and auto-assign staff
                category_id = None
                assigned_to = None
                assigned_at = None
                try:
                    response = supabase.table('request_categories').select('*').execute()
                    categories = response.data if response.data else []
                    text_lower = recognized_text.lower()
                    for category in categories:
                        keywords = category.get('keywords', [])
                        if any(keyword.lower() in text_lower for keyword in keywords):
                            category_id = category['id']
                            break
                    if category_id:
                        cat_response = supabase.table('request_categories').select('*').eq('id', category_id).single().execute()
                        if cat_response.data:
                            required_role = cat_response.data['assigned_role']
                            staff_response = supabase.table('staff_users').select('*').eq('role', required_role).eq('is_active', True).execute()
                            if staff_response.data:
                                assigned_to = staff_response.data[0]['id']
                                assigned_at = datetime.utcnow().isoformat()
                except Exception as e:
                    logger.error(f"Error in category/assignment: {e}")
                # Ensure all fields are strings for save_request_to_db
                category_id = category_id or ""
                assigned_to = assigned_to or ""
                assigned_at = assigned_at or ""
                await self.save_request_to_db(
                    user_data['id'], recognized_text, request_type, priority, category_id, assigned_to, assigned_at
                )
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

# Initialize voice assistant
voice_assistant = VoiceAssistant()

@app.get("/")
async def root():
    return {"message": "ConciAI Voice Assistant is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/voice/process")
async def process_voice(room_number: str, text_input: str = None):
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

# Add all endpoints from oldapp.py
@app.get("/test")
async def test_client():
    return FileResponse("test_client.html")

@app.get("/realtime")
async def realtime_client():
    return FileResponse("realtime_client.html")

@app.get("/simple")
async def simple_client():
    return FileResponse("simple_client.html")

@app.get("/webspeech")
async def webspeech_client():
    return FileResponse("webspeech_client.html")

@app.get("/conversation")
async def conversation_client():
    return FileResponse("conversation_client.html")

@app.get("/test-ai")
async def test_ai():
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
    await websocket.accept()
    logger.info("WebSocket connection established")
    try:
        while True:
            message = await websocket.receive_json()
            room_number = message.get("room_number", "101")
            audio_data = message.get("audio_data")
            text_input = message.get("text_input")
            try:
                if text_input:
                    result = await voice_assistant.process_voice_input_text(text_input, room_number)
                elif audio_data:
                    result = await voice_assistant.process_voice_input_audio(audio_data, room_number)
                else:
                    result = {
                        "error": "No audio_data or text_input provided",
                        "status": "error"
                    }
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