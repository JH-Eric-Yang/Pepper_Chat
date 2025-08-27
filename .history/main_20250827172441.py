#!/usr/bin/env python3
"""
Pepper Robot Voice Interaction System - Main Script (Python 3)
Handles microphone input, Whisper processing, ChatGPT API, and communicates with Python 2 NAOqi bridge
"""

import os
import sys
import json
import subprocess
import threading
import queue
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Third-party imports (to be installed)
try:
    import openai
    import pyaudio
    import numpy as np
    import wave
    import tempfile
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install openai pyaudio numpy")
    sys.exit(1)

# Load configuration from file
def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config.json: {e}")
        # Fallback configuration
        return {
            "robot": {"ip": "192.168.1.100", "port": 9559},
            "openai": {"api_key": "", "model": "gpt-3.5-turbo"},
            "audio": {
                "sample_rate": 16000,
                "chunk_size": 1024,
                "channels": 1,
                "format": "paInt16",
                "silence_threshold": 500,
                "silence_duration": 2.0
            },
            "paths": {
                "python2_executable": r"E:\Project\Robot\Python27\python.exe"
            }
        }

class AudioRecorder:
    """Handles microphone input and audio recording"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config["audio"]
        # Convert format string to pyaudio constant
        format_map = {
            "paInt16": pyaudio.paInt16,
            "paInt32": pyaudio.paInt32,
            "paFloat32": pyaudio.paFloat32
        }
        if isinstance(self.config.get("format"), str):
            self.config["format"] = format_map.get(self.config["format"], pyaudio.paInt16)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.preferred_device_id = self._find_dji_mic_mini()
        
    def _find_dji_mic_mini(self) -> Optional[int]:
        """Find DJI MIC MINI audio device and return its device ID"""
        try:
            device_count = self.audio.get_device_count()
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                device_name = device_info.get('name', '').lower()
                
                # Check for DJI MIC MINI variations in device name
                if ('dji' in device_name and 'mic' in device_name and 'mini' in device_name) or \
                   'dji mic mini' in device_name:
                    if device_info.get('maxInputChannels', 0) > 0:
                        print(f"Found DJI MIC MINI: {device_info['name']} (Device ID: {i})")
                        return i
                        
            print("DJI MIC MINI not found, will use default audio device")
            return None
            
        except Exception as e:
            logging.error(f"Error finding DJI MIC MINI device: {e}")
            return None
            
    def list_audio_devices(self):
        """List all available audio input devices for debugging"""
        try:
            device_count = self.audio.get_device_count()
            print("\nAvailable audio input devices:")
            print("-" * 60)
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    print(f"Device {i}: {device_info['name']} "
                          f"(Channels: {device_info['maxInputChannels']}, "
                          f"Sample Rate: {device_info['defaultSampleRate']})")
            print("-" * 60)
        except Exception as e:
            logging.error(f"Error listing audio devices: {e}")
        
    def start_recording(self) -> bool:
        """Start recording audio from microphone"""
        try:
            # Try to use DJI MIC MINI if available, otherwise use default device
            input_device_index = self.preferred_device_id
            
            if input_device_index is not None:
                print(f"Using DJI MIC MINI (Device ID: {input_device_index})")
            else:
                print("Using default audio input device")
                
            self.stream = self.audio.open(
                format=self.config["format"],
                channels=self.config["channels"],
                rate=self.config["sample_rate"],
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.config["chunk_size"]
            )
            self.is_recording = True
            return True
        except Exception as e:
            # If DJI MIC MINI fails, try with default device
            if self.preferred_device_id is not None:
                logging.warning(f"Failed to use DJI MIC MINI, falling back to default device: {e}")
                try:
                    self.stream = self.audio.open(
                        format=self.config["format"],
                        channels=self.config["channels"],
                        rate=self.config["sample_rate"],
                        input=True,
                        frames_per_buffer=self.config["chunk_size"]
                    )
                    self.is_recording = True
                    print("Successfully fell back to default audio device")
                    return True
                except Exception as fallback_error:
                    logging.error(f"Failed to start recording with fallback device: {fallback_error}")
                    return False
            else:
                logging.error(f"Failed to start recording: {e}")
                return False
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def record_audio(self) -> Optional[str]:
        """Record audio until silence is detected and save to temporary WAV file"""
        if not self.start_recording():
            return None
        
        frames = []
        silent_chunks = 0
        silence_threshold = self.config["silence_threshold"]
        silence_duration = self.config["silence_duration"]
        chunks_per_second = self.config["sample_rate"] / self.config["chunk_size"]
        max_silent_chunks = int(silence_duration * chunks_per_second)
        
        print("Recording... (speak now)")
        
        try:
            while self.is_recording:
                data = self.stream.read(self.config["chunk_size"])
                frames.append(data)
                
                # Check for silence
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                
                if volume < silence_threshold:
                    silent_chunks += 1
                    if silent_chunks > max_silent_chunks:
                        print("Silence detected, stopping recording...")
                        break
                else:
                    silent_chunks = 0
                    
        except Exception as e:
            logging.error(f"Error during recording: {e}")
            return None
        finally:
            self.stop_recording()
        
        if frames:
            # Save audio to temporary WAV file
            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_filename = temp_file.name
                temp_file.close()
                
                # Write WAV file
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(self.config["channels"])
                    wf.setsampwidth(self.audio.get_sample_size(self.config["format"]))
                    wf.setframerate(self.config["sample_rate"])
                    wf.writeframes(b''.join(frames))
                
                return temp_filename
                
            except Exception as e:
                logging.error(f"Error saving audio file: {e}")
                return None
        
        return None
    
    def __del__(self):
        """Cleanup audio resources"""
        if self.stream:
            self.stop_recording()
        self.audio.terminate()


class WhisperProcessor:
    """Handles speech-to-text using OpenAI Whisper API"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        print("Initialized OpenAI Whisper API client")
        
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file to text using OpenAI Whisper API"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            text = transcript.text.strip()
            if text:
                print(f"Transcribed: {text}")
                # Clean up temporary file
                try:
                    os.unlink(audio_file_path)
                except Exception as cleanup_error:
                    logging.warning(f"Could not delete temporary audio file {audio_file_path}: {cleanup_error}")
                return text
            return None
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            # Clean up temporary file on error
            try:
                os.unlink(audio_file_path)
            except Exception as cleanup_error:
                logging.warning(f"Could not delete temporary audio file {audio_file_path}: {cleanup_error}")
            return None


class ChatGPTClient:
    """Handles ChatGPT API interactions"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key not provided")
    
    def get_response(self, user_input: str) -> Optional[str]:
        """Get response from ChatGPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are Pepper, a friendly humanoid robot assistant. Keep responses concise and engaging."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content.strip()
            print(f"ChatGPT response: {reply}")
            return reply
            
        except Exception as e:
            logging.error(f"ChatGPT API error: {e}")
            return "Sorry, I'm having trouble understanding right now."


class NAOqiBridge:
    """Manages communication with Python 2 NAOqi bridge subprocess"""
    
    def __init__(self, robot_ip: str, robot_port: int, python2_path: str = None):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.process = None
        self.python2_path = python2_path or r"E:\Project\Robot\Python27\python.exe"
        self.bridge_script = Path(__file__).parent / "naoqi_bridge.py"
        
    def start_bridge(self) -> bool:
        """Start the Python 2 NAOqi bridge subprocess"""
        try:
            cmd = [
                self.python2_path,
                str(self.bridge_script),
                self.robot_ip,
                str(self.robot_port)
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Wait for bridge to initialize
            time.sleep(2)
            
            if self.process.poll() is None:
                print("NAOqi bridge started successfully")
                return True
            else:
                error = self.process.stderr.read()
                print(f"Bridge failed to start: {error}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to start NAOqi bridge: {e}")
            return False
    
    def send_command(self, command: Dict[str, Any]) -> Optional[str]:
        """Send command to NAOqi bridge and get response"""
        if not self.process or self.process.poll() is not None:
            logging.error("NAOqi bridge is not running")
            return None
        
        try:
            command_json = json.dumps(command) + "\n"
            self.process.stdin.write(command_json)
            self.process.stdin.flush()
            
            # Read response, skipping any log lines that don't start with '{'
            max_attempts = 10  # Prevent infinite loop
            attempts = 0
            
            while attempts < max_attempts:
                response = self.process.stdout.readline().strip()
                attempts += 1
                
                if not response:
                    logging.error("Empty response from bridge")
                    return {"success": False, "error": "Empty response from bridge"}
                
                # Skip NAOqi log lines (they don't start with '{')
                if response.startswith('{'):
                    try:
                        return json.loads(response)
                    except json.JSONDecodeError as json_error:
                        logging.error(f"Invalid JSON response from bridge: '{response}' - {json_error}")
                        return {"success": False, "error": f"Invalid JSON response: {response}"}
                else:
                    # This is a log line, skip it and try the next line
                    logging.debug(f"Skipping NAOqi log line: {response}")
                    continue
            
            logging.error(f"No valid JSON response found after {max_attempts} attempts")
            return {"success": False, "error": "No valid JSON response found"}
            
        except Exception as e:
            logging.error(f"Communication error with bridge: {e}")
        
        return None
    
    def speak(self, text: str, language: str = "English") -> bool:
        """Send text to robot for speech"""
        command = {
            "action": "speak",
            "text": text,
            "language": language
        }
        
        response = self.send_command(command)
        return response and response.get("success", False)
    
    def stop_bridge(self):
        """Stop the NAOqi bridge subprocess"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


class VoiceInteractionSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        # Load configuration
        self.config = load_config()
        
        # Initialize components
        self.recorder = AudioRecorder(self.config)
        self.whisper = WhisperProcessor(self.config["openai"]["api_key"])
        self.chatgpt = ChatGPTClient(self.config["openai"]["api_key"])
        self.bridge = NAOqiBridge(
            self.config["robot"]["ip"], 
            self.config["robot"]["port"],
            self.config["paths"]["python2_executable"]
        )
        
        self.running = False
    
    def start(self):
        """Start the voice interaction system"""
        print("Starting Pepper Voice Interaction System...")
        
        # List available audio devices for debugging
        self.recorder.list_audio_devices()
        
        # Start NAOqi bridge
        if not self.bridge.start_bridge():
            print("Failed to start NAOqi bridge. Exiting.")
            return False
        
        self.running = True
        print("System ready! Press Ctrl+C to stop.")
        
        try:
            while self.running:
                print("\nListening for voice input...")
                
                # Record audio
                audio_file_path = self.recorder.record_audio()
                if audio_file_path is None:
                    continue
                
                # Transcribe speech
                user_text = self.whisper.transcribe(audio_file_path)
                if not user_text:
                    continue
                
                # Get ChatGPT response
                robot_response = self.chatgpt.get_response(user_text)
                if not robot_response:
                    continue
                
                # Send to robot for speech
                if self.bridge.speak(robot_response):
                    print("Response sent to robot successfully")
                else:
                    print("Failed to send response to robot")
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the system"""
        self.running = False
        self.bridge.stop_bridge()
        print("System stopped.")


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load and check configuration
    config = load_config()
    api_key = config["openai"]["api_key"] or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Please set your OpenAI API key in config.json or OPENAI_API_KEY environment variable")
        return
    
    # Create and start system
    system = VoiceInteractionSystem()
    system.start()


if __name__ == "__main__":
    main()