# Pepper Robot Voice Interaction System

A complete voice interaction system for Pepper robot with multi-agent personality support that captures microphone input, processes it with Whisper speech recognition, sends it to ChatGPT for intelligent responses using configurable AI agents, and makes the robot speak using NAOqi text-to-speech.

## System Architecture

The system uses a **subprocess bridge architecture** with **multi-agent AI personality system** and has two main components:

1. **Main Script (Python 3)**: `main.py`
   - Handles microphone input and audio recording with Silero VAD or volume-based detection
   - Processes speech-to-text using OpenAI Whisper
   - Communicates with ChatGPT API for intelligent responses using multiple AI agents
   - Manages the Python 2 NAOqi bridge subprocess
   - Supports dynamic agent switching via voice commands

2. **NAOqi Bridge (Python 2.7)**: `naoqi_bridge.py` 
   - Connects to Pepper robot using NAOqi SDK
   - Handles text-to-speech, posture control, LED control
   - Communicates with main script via JSON messages over stdin/stdout

## Prerequisites

### Hardware
- Pepper robot (connected to same network)
- Computer with microphone
- Network connection between computer and robot

### Software
- Python 3.7+ (for main application)
- Python 2.7 (for NAOqi bridge) - Located at `E:\Project\Robot\Python27\python.exe`
- NAOqi SDK - Located at `E:\Project\Robot\naoqi_sdk`

## Installation

### 1. Install Python 3 Dependencies
```bash
cd E:\Project\Robot\Peper_development
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key
Set your OpenAI API key in one of these ways:

**Option A: Environment Variable**
```bash
set OPENAI_API_KEY=your_api_key_here
```

**Option B: Configuration File**
Edit `config.json` and set your API key:
```json
{
  "openai": {
    "api_key": "your_api_key_here"
  }
}
```

### 3. Configure Robot Connection
Edit `config.json` to set your Pepper robot's IP address:
```json
{
  "robot": {
    "ip": "192.168.1.100",
    "port": 9559
  }
}
```

## Multi-Agent Personality System

The system features **6 different AI agents** with unique personalities and communication styles:

### Available Agents
1. **Friendly Science Communicator** (default)
   - Approachable, upbeat, and enthusiastic about technology
   - Explains concepts in clear, everyday language with relatable examples

2. **Playful Storyteller**
   - Creative, witty, and dramatic in a fun way
   - Turns information into short stories and "what if" adventures

3. **Curious Explorer**
   - Adventurous, curious, and collaborative
   - Works alongside students to figure things out together

4. **Digital Safety Guide**
   - Trustworthy, supportive, and practical
   - Helps with online safety, privacy, and digital citizenship

5. **Quiz Master**
   - Lively, energetic, and encouraging
   - Loves asking quick quizzes and making learning feel like a game

6. **Warm Mentor**
   - Calm, encouraging, and respectful
   - Treats young people as capable learners and builds confidence

### Agent Voice Commands
- **"list agents"** - Show all available agents
- **"switch to [agent_name]"** - Change to a different agent
- **"current agent"** - Show current agent information

## Usage

### Basic Usage
```bash
python main.py
```

The system will:
1. Start the NAOqi bridge subprocess
2. Display current agent and available voice commands
3. Begin listening for voice input with speech detection
4. Process speech with Whisper
5. Get response from ChatGPT using current agent personality
6. Send response to robot for speech synthesis

### Voice Interaction Flow
1. **Listen**: System captures audio using advanced speech detection (VAD or volume-based)
2. **Transcribe**: Whisper converts speech to text
3. **Process**: ChatGPT generates intelligent response using current agent personality
4. **Speak**: Robot speaks the response using NAOqi TTS
5. **Agent Management**: Handle voice commands for switching between personalities

## Configuration

### Audio Settings
Adjust microphone sensitivity and recording parameters in `config.json`:
```json
{
  "audio": {
    "silence_threshold": 500,
    "silence_duration": 2.0,
    "sample_rate": 16000
  }
}
```

### Robot Behavior
Customize robot behavior:
```json
{
  "behavior": {
    "default_posture": "Stand",
    "eye_color_on_listening": "blue",
    "eye_color_on_speaking": "green"
  }
}
```

### Speech Settings
Configure text-to-speech:
```json
{
  "speech": {
    "default_language": "English",
    "volume": 0.8,
    "speed": 1.0
  }
}
```

## Available Commands

The NAOqi bridge supports these commands:

### Text-to-Speech
```python
{"action": "speak", "text": "Hello world", "language": "English"}
```

### Robot Control
```python
{"action": "set_posture", "posture": "Stand"}
{"action": "set_eye_color", "color": "blue"}
{"action": "move_head", "yaw": 0.0, "pitch": 0.1}
```

### Audio Control
```python
{"action": "set_volume", "volume": 0.8}
{"action": "set_speed", "speed": 1.2}
```

## Troubleshooting

### Common Issues

**1. NAOqi Bridge Fails to Start**
- Check Python 2.7 path in `config.json`
- Verify NAOqi SDK is installed at specified path
- Ensure robot IP is correct and accessible

**2. Audio Recording Issues**
- Check microphone permissions
- Adjust `silence_threshold` in config
- Install proper audio drivers

**3. OpenAI API Errors**
- Verify API key is valid
- Check internet connection
- Monitor API usage limits

**4. Robot Connection Issues**
- Ping robot IP to verify network connectivity
- Check robot is powered on and connected to network
- Verify NAOqi version compatibility

### Testing Individual Components

**Test NAOqi Bridge Only:**
```bash
E:\Project\Robot\Python27\python.exe naoqi_bridge.py 192.168.1.100 9559
```

Then send JSON commands via stdin:
```json
{"action": "speak", "text": "Test message"}
```

**Test Audio Recording:**
Modify `main.py` to only test microphone input without other components.

## Project Structure

```
E:\Project\Robot\Peper_development\
├── main.py              # Python 3 main application with multi-agent system
├── naoqi_bridge.py      # Python 2 NAOqi bridge for robot communication
├── config.json          # Configuration file with agent definitions
├── requirements.txt     # Python 3 dependencies
├── agent_manager.py     # Agent management utilities
├── agent_personality.md # Agent personality documentation
├── diagnostic_test.py   # System diagnostic testing
├── debug_tts.py         # TTS debugging utilities
├── simple_test.py       # Simple system testing
├── test_bridge.py       # NAOqi bridge testing
├── test_components.py   # Component testing utilities
└── README.md           # This documentation

E:\Project\Robot\Python27\    # Python 2.7 installation
E:\Project\Robot\naoqi_sdk\   # NAOqi SDK
```

## Development Notes

- The system uses subprocess communication with JSON messages
- Advanced speech detection using Silero VAD with volume-based fallback
- Multi-agent personality system with 6 configurable AI agents
- Robot responses are limited to 350 tokens for natural speech
- Error handling includes automatic retries and graceful degradation
- Agent switching is persistent across sessions via config.json
- Support for both DJI MIC MINI and default audio devices

## License

This project is for educational and development purposes.