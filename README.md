# ConciAI Voice Assistant Backend

This is the Python backend for the ConciAI hotel voice assistant system.

## Features

- Voice recognition and processing
- Gemini AI integration for intelligent responses
- Automatic request/complaint detection
- Real-time database integration with Supabase
- WebSocket support for ESP32 communication
- RESTful API for testing and integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

3. Start the server:
```bash
python app.py
```

## API Endpoints

- `GET /` - Health check
- `POST /voice/process` - Process voice/text input
- `GET /users/{room_number}` - Get user by room number
- `WebSocket /ws/voice` - Real-time voice communication

## ESP32 Integration

The system supports ESP32 microcontrollers with voice activation using "Hey Conci" wake word.

## Testing

You can test the system by sending POST requests to `/voice/process` with text input for development purposes.