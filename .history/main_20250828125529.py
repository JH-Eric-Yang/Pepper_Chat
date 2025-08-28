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
from enum import Enum

# Third-party imports (to be installed)
try:
    import openai
    import pyaudio
    import numpy as np
    import wave
    import tempfile
    import torch
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install openai pyaudio numpy silero-vad torch torchaudio")
    sys.exit(1)pi

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
                "silence_duration": 3.0
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
        
        # Initialize Silero VAD
        try:
            self.vad_model = load_silero_vad()
            self.vad_iterator = VADIterator(
                model=self.vad_model,
                threshold=0.5,
                sampling_rate=self.config["sample_rate"],
                min_silence_duration_ms=500,  # 0.5 second silence to end
                speech_pad_ms=30  # 30ms padding around speech
            )
            print("Silero VAD initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize VAD, falling back to volume detection: {e}")
            self.vad_model = None
            self.vad_iterator = None
        
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
    """Handles ChatGPT API interactions with multi-agent support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = openai.OpenAI(api_key=config["openai"]["api_key"] or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.current_agent = config["agents"]["current_agent"]
        self.available_agents = config["agents"]["available_agents"]
        
    def get_current_agent_info(self) -> Dict[str, str]:
        """Get current agent information"""
        if self.current_agent in self.available_agents:
            return self.available_agents[self.current_agent]
        else:
            # Fallback to first available agent if current agent not found
            first_agent = next(iter(self.available_agents.keys()))
            self.current_agent = first_agent
            return self.available_agents[first_agent]
    
    def switch_agent(self, agent_key: str) -> bool:
        """Switch to a different agent"""
        if agent_key in self.available_agents:
            self.current_agent = agent_key
            agent_info = self.available_agents[agent_key]
            print(f"Switched to agent: {agent_info['name']} - {agent_info['description']}")
            return True
        else:
            print(f"Agent '{agent_key}' not found. Available agents: {list(self.available_agents.keys())}")
            return False
    
    def list_agents(self):
        """List all available agents"""
        print("\nAvailable Agents:")
        print("-" * 50)
        for key, agent in self.available_agents.items():
            marker = " [CURRENT]" if key == self.current_agent else ""
            print(f"{key}: {agent['name']}{marker}")
            print(f"  Description: {agent['description']}")
            print()
    
    def get_response(self, user_input: str) -> Optional[str]:
        """Get response from ChatGPT using current agent"""
        try:
            agent_info = self.get_current_agent_info()
            
            response = self.client.chat.completions.create(
                model=self.config["openai"]["model"],
                messages=[
                    {"role": "system", "content": agent_info["system_prompt"]},
                    {"role": "user", "content": user_input}
                ],
                max_completion_tokens=self.config["openai"]["max_completion_tokens"],
                temperature=self.config["openai"]["temperature"]
            )
            
            reply = response.choices[0].message.content.strip()
            print(f"[{agent_info['name']}] Response: {reply}")
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
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for bridge to initialize (longer wait like test_bridge.py)
            time.sleep(3)
            
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
    
    def test_audio(self) -> bool:
        """Test audio system before attempting speech"""
        command = {"action": "test_audio"}
        response = self.send_command(command)
        if response and response.get("success", False):
            print("Audio system test passed")
            return True
        else:
            error_msg = response.get("error", "Unknown error") if response else "No response from bridge"
            logging.error(f"Audio test failed: {error_msg}")
            return False
    
    def check_speaking_status(self) -> bool:
        """Check if robot is currently speaking"""
        if not self.process or self.process.poll() is not None:
            return False
        
        command = {"action": "check_speaking"}
        response = self.send_command(command)
        if response and response.get("success", False):
            return response.get("is_speaking", False)
        return False
    
    def wait_for_speech_completion(self, timeout: float = 30.0) -> bool:
        """Wait for robot to finish speaking"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.check_speaking_status():
                return True
            time.sleep(0.1)  # Check every 100ms
        
        logging.warning("Speech completion wait timed out")
        return False
    
    def speak(self, text: str, language: str = "English") -> bool:
        """Send text to robot for speech and wait for completion"""
        # Audio testing removed - speak directly without testing

        command = {
            "action": "speak",
            "text": text,
            "language": language
        }

        print(command)
        
        response = self.send_command(command)
        if response and response.get("success", False):
            # Speech command was successful and should be completed
            return True
        else:
            # Log the error if available
            error_msg = response.get("error", "Unknown error") if response else "No response from bridge"
            logging.error(f"Speech failed: {error_msg}")
            return False
    
    def stop_bridge(self):
        """Stop the NAOqi bridge subprocess"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


class SystemState(Enum):
    """System states for voice interaction"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"

class VoiceInteractionSystem:
    """Main system orchestrator with multi-agent support"""
    
    def __init__(self):
        # Load configuration
        self.config = load_config()
        
        # Initialize components
        self.recorder = AudioRecorder(self.config)
        self.whisper = WhisperProcessor(self.config["openai"]["api_key"])
        self.chatgpt = ChatGPTClient(self.config)
        self.bridge = NAOqiBridge(
            self.config["robot"]["ip"], 
            self.config["robot"]["port"],
            self.config["paths"]["python2_executable"]
        )
        
        self.running = False
        self.state = SystemState.IDLE
        self.state_lock = threading.Lock()
    
    def switch_agent(self, agent_key: str) -> bool:
        """Switch to a different agent and update config"""
        if self.chatgpt.switch_agent(agent_key):
            self.config["agents"]["current_agent"] = agent_key
            # Save updated config
            self.save_config()
            return True
        return False
    
    def list_agents(self):
        """List all available agents"""
        self.chatgpt.list_agents()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_path = Path(__file__).parent / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def handle_system_commands(self, user_text: str) -> bool:
        """Handle system commands like agent switching"""
        user_text_lower = user_text.lower().strip()
        
        # Agent switching commands
        if user_text_lower.startswith("switch to ") or user_text_lower.startswith("change to "):
            agent_name = user_text_lower.replace("switch to ", "").replace("change to ", "").strip()
            agent_key = agent_name.replace(" ", "_").lower()
            if self.switch_agent(agent_key):
                return True
        
        # List agents command
        elif "list agents" in user_text_lower or "show agents" in user_text_lower or "available agents" in user_text_lower:
            self.list_agents()
            return True
        
        # Current agent command
        elif "current agent" in user_text_lower or "which agent" in user_text_lower:
            agent_info = self.chatgpt.get_current_agent_info()
            print(f"Current agent: {agent_info['name']} - {agent_info['description']}")
            return True
        
        return False
    
    def set_state(self, new_state: SystemState):
        """Thread-safe state change"""
        with self.state_lock:
            if self.state != new_state:
                print(f"State: {self.state.value} -> {new_state.value}")
                self.state = new_state
    
    def get_state(self) -> SystemState:
        """Thread-safe state getter"""
        with self.state_lock:
            return self.state
    
    def should_start_listening(self) -> bool:
        """Check if system should start listening for input"""
        return self.get_state() == SystemState.IDLE
    
    def wait_for_speech_detection(self) -> Optional[str]:
        """Wait for speech detection using Silero VAD or fallback to volume detection"""
        if not self.recorder.start_recording():
            return None
        
        # Use VAD if available, otherwise fallback to volume detection
        if self.recorder.vad_iterator:
            return self._vad_speech_detection()
        else:
            return self._volume_speech_detection()
    
    def _vad_speech_detection(self) -> Optional[str]:
        """Speech detection using Silero VAD"""
        frames = []
        speech_started = False
        audio_buffer = np.array([], dtype=np.int16)
        vad_chunk_size = 512  # Required by Silero VAD for 16kHz
        
        print("Waiting for speech (VAD)...")
        
        try:
            while self.recorder.is_recording and self.get_state() == SystemState.LISTENING:
                data = self.recorder.stream.read(self.config["audio"]["chunk_size"])
                
                # Convert to int16 and add to buffer
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                # Process in VAD-sized chunks
                while len(audio_buffer) >= vad_chunk_size:
                    # Extract VAD chunk
                    vad_chunk = audio_buffer[:vad_chunk_size]
                    audio_buffer = audio_buffer[vad_chunk_size:]
                    
                    # Convert to float32 for VAD
                    audio_float = vad_chunk.astype(np.float32) / 32768.0
                    audio_tensor = torch.from_numpy(audio_float)
                    
                    # Get VAD decision
                    vad_result = self.recorder.vad_iterator(audio_tensor)
                    
                    if vad_result is not None:  # Speech event detected
                        if 'start' in vad_result:  # Speech started
                            if not speech_started:
                                speech_started = True
                                print("Speech detected (VAD), recording...")
                        elif 'end' in vad_result:  # Speech ended
                            if speech_started:
                                print("End of speech detected (VAD)")
                                frames.append(data)  # Include final frame
                                return self._save_audio_frames(frames) if frames else None
                
                # Always collect frames during recording
                frames.append(data)
                
        except Exception as e:
            logging.error(f"Error during VAD speech detection: {e}")
            return None
        finally:
            self.recorder.stop_recording()
            # Reset VAD state for next detection
            if self.recorder.vad_iterator:
                self.recorder.vad_iterator.reset_states()
        
        return self._save_audio_frames(frames) if frames and speech_started else None
    
    def _volume_speech_detection(self) -> Optional[str]:
        """Fallback volume-based speech detection"""
        frames = []
        silent_chunks = 0
        speech_started = False
        silence_threshold = self.config["audio"]["silence_threshold"]
        silence_duration = self.config["audio"]["silence_duration"]
        chunks_per_second = self.config["audio"]["sample_rate"] / self.config["audio"]["chunk_size"]
        max_silent_chunks = int(silence_duration * chunks_per_second)
        
        # Minimum speech duration (0.5 seconds)
        min_speech_chunks = int(0.5 * chunks_per_second)
        speech_chunks = 0
        
        print("Waiting for speech (volume)...")
        
        try:
            while self.recorder.is_recording and self.get_state() == SystemState.LISTENING:
                data = self.recorder.stream.read(self.config["audio"]["chunk_size"])
                
                # Check for speech
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                
                if volume > silence_threshold:
                    if not speech_started:
                        speech_started = True
                        print("Speech detected (volume), recording...")
                    
                    frames.append(data)
                    speech_chunks += 1
                    silent_chunks = 0
                else:
                    if speech_started:
                        frames.append(data)
                        silent_chunks += 1
                        
                        # End recording if we have enough speech and silence
                        if (speech_chunks >= min_speech_chunks and 
                            silent_chunks > max_silent_chunks):
                            print("End of speech detected (volume)")
                            break
                    else:
                        # Still waiting for speech to start
                        continue
                        
        except Exception as e:
            logging.error(f"Error during volume speech detection: {e}")
            return None
        finally:
            self.recorder.stop_recording()
        
        # Only process if we have enough speech
        if frames and speech_chunks >= min_speech_chunks:
            return self._save_audio_frames(frames)
        
        return None
    
    def _save_audio_frames(self, frames) -> Optional[str]:
        """Save audio frames to temporary WAV file"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_filename = temp_file.name
            temp_file.close()
            
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.config["audio"]["channels"])
                wf.setsampwidth(self.recorder.audio.get_sample_size(self.config["audio"]["format"]))
                wf.setframerate(self.config["audio"]["sample_rate"])
                wf.writeframes(b''.join(frames))
            
            return temp_filename
            
        except Exception as e:
            logging.error(f"Error saving audio file: {e}")
            return None
    
    def start(self):
        """Start the voice interaction system"""
        print("Starting Pepper Voice Interaction System...")
        
        # Show current agent information
        agent_info = self.chatgpt.get_current_agent_info()
        print(f"\nCurrent Agent: {agent_info['name']}")
        print(f"Description: {agent_info['description']}")
        print(f"\nVoice Commands:")
        print("- 'list agents' - Show available agents")
        print("- 'switch to [agent_name]' - Change agent")
        print("- 'current agent' - Show current agent info")
        print("-" * 50)
        
        # List available audio devices for debugging
        self.recorder.list_audio_devices()
        
        # Start NAOqi bridge
        if not self.bridge.start_bridge():
            print("Failed to start NAOqi bridge. Exiting.")
            return False
        
        self.running = True
        self.set_state(SystemState.IDLE)
        print("System ready! Press Ctrl+C to stop.")
        
        try:
            while self.running:
                current_state = self.get_state()
                
                if current_state == SystemState.IDLE:
                    print("\nReady for interaction...")
                    self.set_state(SystemState.LISTENING)
                
                elif current_state == SystemState.LISTENING:
                    # Wait for speech detection
                    audio_file_path = self.wait_for_speech_detection()
                    if audio_file_path:
                        self.set_state(SystemState.PROCESSING)
                        
                        # Transcribe speech
                        user_text = self.whisper.transcribe(audio_file_path)
                        if user_text:
                            # Check if it's a system command first
                            if self.handle_system_commands(user_text):
                                # System command handled, return to idle
                                self.set_state(SystemState.IDLE)
                                time.sleep(1)
                            else:
                                # Get ChatGPT response using current agent
                                robot_response = self.chatgpt.get_response(user_text)
                                if robot_response:
                                    self.set_state(SystemState.SPEAKING)
                                    
                                    # Send to robot for speech
                                    print("Robot speaking...")
                                    speech_result = self.bridge.speak(robot_response)
                                    if speech_result:
                                        # Wait for robot to actually finish speaking
                                        print("Waiting for robot to finish speaking...")
                                        if self.bridge.wait_for_speech_completion(timeout=30.0):
                                            print("Robot finished speaking")
                                        else:
                                            print("Speech completion wait timed out")
                                        # Small additional buffer to ensure clean audio separation
                                        time.sleep(0.5)
                                    else:
                                        print("Failed to send response to robot")
                                    
                                    # Return to idle state
                                    self.set_state(SystemState.IDLE)
                                    time.sleep(0.5)  # Brief pause before next interaction
                                else:
                                    self.set_state(SystemState.IDLE)
                        else:
                            self.set_state(SystemState.IDLE)
                    else:
                        # No valid speech detected, back to listening
                        continue
                
                elif current_state in [SystemState.PROCESSING, SystemState.SPEAKING]:
                    # Wait for processing/speaking to complete
                    time.sleep(0.1)
                
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