#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAOqi Bridge Script (Python 2.7) - Minimal Version
Handles text-to-speech communication with Pepper robot using NAOqi SDK
Communicates with main Python 3 script via stdin/stdout JSON messages
"""

import sys
import os
import json
import time
import re
import unicodedata

# Add NAOqi SDK to Python path - load from config
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    naoqi_sdk_path = config.get('paths', {}).get('naoqi_sdk_path', r"C:\Users\ericy\Documents\naoqi_sdk")
except Exception:
    # Fallback to your actual path if config reading fails
    naoqi_sdk_path = r"C:\Users\ericy\Documents\naoqi_sdk"

naoqi_lib_path = os.path.join(naoqi_sdk_path, 'lib')
sys.path.append(naoqi_lib_path)

# Set environment variables to suppress NAOqi logs before importing
os.environ["NAOQI_LOG_LEVEL"] = "SILENT"
os.environ["ALDEBARAN_LOG_LEVEL"] = "SILENT"

try:
    import naoqi
    from naoqi import ALProxy, ALBroker, ALModule
except ImportError as e:
    print(json.dumps({"error": "NAOqi import failed", "details": str(e)}))
    sys.exit(1)


class PepperBridge:
    """Minimal bridge class for Pepper robot text-to-speech communication"""
    
    def __init__(self, robot_ip, robot_port):
        self.robot_ip = robot_ip
        self.robot_port = int(robot_port)
        self.tts = None
        self.animated_speech = None
        self.audio_device = None
        self.connected = False
        self.speaking = False
        self.microphones_disabled = False
        
        # Connect to robot
        self.connect()
    
    def connect(self):
        """Connect to Pepper robot"""
        try:
            # Initialize text-to-speech proxy
            self.tts = ALProxy("ALTextToSpeech", self.robot_ip, self.robot_port)
            
            # Initialize animated speech proxy (better for NAOqi 2.5)
            try:
                self.animated_speech = ALProxy("ALAnimatedSpeech", self.robot_ip, self.robot_port)
            except Exception as anim_ex:
                self.animated_speech = None
            
            # Initialize audio device proxy
            try:
                self.audio_device = ALProxy("ALAudioDevice", self.robot_ip, self.robot_port)
            except Exception as audio_ex:
                self.audio_device = None
            
            # NAOqi 2.5 specific initialization
            # Reset TTS parameters to ensure clean state
            try:
                self.tts.resetSpeed()
                self.tts.resetVolume()
            except Exception:
                pass  # Methods may not exist in all versions
            
            # Set TTS volume (NAOqi 2.5 prefers values between 0.0-1.0)
            self.tts.setVolume(0.9)
            
            # Set master volume using ALAudioDevice (more reliable for NAOqi 2.5)
            if self.audio_device:
                try:
                    # Set output volume to maximum for NAOqi 2.5
                    self.audio_device.setOutputVolume(95)  # 0-100 range
                    
                    # Unmute audio output if muted
                    try:
                        if self.audio_device.isOutputMuted():
                            self.audio_device.muteAudioOut(False)
                    except Exception as mute_error:
                        pass  # Method may not exist in this NAOqi version
                        
                    # Enable audio output (NAOqi 2.5 specific)
                    try:
                        self.audio_device.enableAudioOut(True)
                    except Exception:
                        pass  # Method may not exist
                        
                except Exception as audio_setup_error:
                    # Audio device setup failed, but continue
                    pass
            
            # For NAOqi 2.5, ensure TTS is properly initialized
            available_voices = self.tts.getAvailableVoices()
            available_languages = self.tts.getAvailableLanguages()
            
            # Set English as default language for NAOqi 2.5
            try:
                if "English" in available_languages:
                    self.tts.setLanguage("English")
                elif available_languages:
                    self.tts.setLanguage(available_languages[0])
            except Exception:
                pass
            
            # Set default voice if available (NAOqi 2.5 compatibility)
            if available_voices:
                try:
                    # Prefer English voices for NAOqi 2.5
                    english_voices = [v for v in available_voices if 'naoenu' in v.lower() or 'english' in v.lower()]
                    if english_voices:
                        self.tts.setVoice(english_voices[0])
                    else:
                        self.tts.setVoice(available_voices[0])
                except Exception:
                    pass
            
            # Set speech parameters for NAOqi 2.5
            try:
                self.tts.setParameter("speed", 100)  # Normal speed for NAOqi 2.5
                self.tts.setParameter("pitchShift", 1.0)  # Normal pitch
            except Exception:
                pass
            
            self.connected = True
            
            # Disable robot microphones immediately after connection to prevent feedback
            if self.disable_robot_microphones():
                self.send_response({"success": True, "message": "Connected to Pepper robot, microphones disabled"})
            else:
                self.send_response({"success": True, "message": "Connected to Pepper robot, warning: failed to disable microphones"})
            
        except Exception as e:
            self.connected = False
            self.send_response({"success": False, "error": str(e)})
    
    def send_response(self, response):
        """Send JSON response to main script"""
        try:
            print(json.dumps(response, ensure_ascii=True))
            sys.stdout.flush()
        except Exception as e:
            # Fallback error response
            print(json.dumps({"success": False, "error": "Response encoding failed"}, ensure_ascii=True))
            sys.stdout.flush()
    
    def clean_text_for_speech(self, text):
        """Clean text for speech - simplified version to avoid corruption"""
        # For basic ASCII text like "Another Test", just return it as-is
        # Only do minimal cleaning to avoid corruption
        
        # If it's already a clean ASCII string, don't process it
        if isinstance(text, str) and all(ord(c) < 128 for c in text):
            # Simple ASCII text, just clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # For non-ASCII or problematic text, do minimal safe cleaning
        if isinstance(text, str):
            # In Python 2.7, this might be a byte string
            try:
                # Try to decode if it's bytes
                text = text.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                # Either already unicode or can't decode, continue
                pass
        
        # Replace only the most common problematic characters
        replacements = {
            u'\u2019': "'",  # Right single quotation mark
            u'\u2018': "'",  # Left single quotation mark  
            u'\u201c': '"',  # Left double quotation mark
            u'\u201d': '"',  # Right double quotation mark
            u'\u2013': '-',  # En dash
            u'\u2014': '-',  # Em dash
            u'\u2026': '...' # Horizontal ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def speak(self, text, language="English"):
        """Make robot speak the given text"""
        if not self.connected:
            self.send_response({"success": False, "error": "Not connected to robot"})
            return
        
        # Wait for any ongoing speech to complete
        if self.speaking:
            time.sleep(1)  # Brief wait if still speaking
            
        self.speaking = True
        microphones_were_disabled = False
        
        try:
            # Disable robot microphones before speaking to prevent feedback
            if self.disable_robot_microphones():
                microphones_were_disabled = True
            
            # Clean text for speech
            clean_text = self.clean_text_for_speech(text)
            
            # Debug removed - text cleaning working
            
            # Check audio volume first
            current_volume = self.tts.getVolume()
            # Set language
            if language.lower() == "english":
                self.tts.setLanguage("English")
            elif language.lower() == "french":
                self.tts.setLanguage("French")
            elif language.lower() == "japanese":
                self.tts.setLanguage("Japanese")
            elif language.lower() == "chinese":
                self.tts.setLanguage("Chinese")
            elif language.lower() == "german":
                self.tts.setLanguage("German")
            elif language.lower() == "spanish":
                self.tts.setLanguage("Spanish")
            else:
                self.tts.setLanguage("English")  # Default fallback
            
            current_language = self.tts.getLanguage()
            
            # Ensure volume is sufficient
            if current_volume < 0.5:
                self.tts.setVolume(0.8)
                current_volume = 0.8
            
            # NAOqi 2.5 specific speech handling
            # First ensure audio is ready
            if self.audio_device:
                try:
                    # Check and fix audio state for NAOqi 2.5
                    if self.audio_device.isOutputMuted():
                        self.audio_device.muteAudioOut(False)
                    
                    current_master_volume = self.audio_device.getOutputVolume()
                    if current_master_volume < 80:
                        self.audio_device.setOutputVolume(95)
                except Exception:
                    pass
            
            # Speak the text - try multiple methods for NAOqi 2.5 compatibility
            speech_success = False
            
            # Method 1: Try ALAnimatedSpeech first (recommended for NAOqi 2.5)
            error_messages = []
            if self.animated_speech and not speech_success:
                try:
                    # Use ALAnimatedSpeech for better speech quality in NAOqi 2.5
                    self.animated_speech.say(clean_text)
                    speech_success = True
                except Exception as anim_error:
                    error_messages.append("ALAnimatedSpeech failed: " + str(anim_error))
            
            # Method 2: Fallback to regular TTS if AnimatedSpeech failed
            if not speech_success:
                try:
                    self.tts.say(clean_text)
                    speech_success = True
                except Exception as tts_error:
                    error_messages.append("TTS failed: " + str(tts_error))
            
            if not speech_success:
                all_errors = " | ".join(error_messages)
                raise Exception("All speech methods failed: " + all_errors)
            
            # Add a small delay after speech to ensure it's fully complete
            time.sleep(0.5)
            
            self.send_response({
                "success": True, 
                "message": "Speech completed", 
                "volume": current_volume, 
                "language": current_language,
                "microphones_disabled": microphones_were_disabled
            })
            
        except Exception as e:
            self.send_response({"success": False, "error": str(e)})
        finally:
            # Keep robot microphones disabled - don't re-enable after speech
            # This prevents audio feedback since we want them disabled throughout the session
            self.speaking = False
    
    def set_volume(self, volume):
        """Set speech volume (0.0 to 1.0)"""
        if not self.connected:
            self.send_response({"success": False, "error": "Not connected to robot"})
            return
        
        try:
            self.tts.setVolume(max(0.0, min(1.0, float(volume))))
            self.send_response({"success": True, "message": "Volume set"})
        except Exception as e:
            self.send_response({"success": False, "error": str(e)})
    
    def set_speech_speed(self, speed):
        """Set speech speed (0.5 to 4.0)"""
        if not self.connected:
            self.send_response({"success": False, "error": "Not connected to robot"})
            return
        
        try:
            self.tts.setParameter("speed", max(0.5, min(4.0, float(speed))))
            self.send_response({"success": True, "message": "Speech speed set"})
        except Exception as e:
            self.send_response({"success": False, "error": str(e)})
    
    def disable_robot_microphones(self):
        """Disable robot's microphones during speech output"""
        if not self.connected or not self.audio_device:
            return False
        
        try:
            # Disable input by setting the microphone gain to 0
            self.audio_device.setOutputVolume(0)  
            self.audio_device.setParameter("MicOn", 0)
            self.microphones_disabled = True
            return True
        except Exception as e:
            print('Microphone disable error:')
            print(e)
            return False
    
    def enable_robot_microphones(self):
        """Re-enable robot's microphones after speech"""
        if not self.connected or not self.audio_device:
            return False
        
        try:
            # Re-enable microphones by restoring volume and mic parameter
            
            self.audio_device.enableAudioOut(True)
            self.audio_device.setOutputVolume(95)  
            self.audio_device.setParameter("MicOn", 1)
            self.microphones_disabled = False
            return True
        except Exception as e:
            return False
    
    def mute_robot_microphones(self, mute=True):
        """Alternative method: Mute/unmute robot's microphone input volume"""
        if not self.connected or not self.audio_device:
            return False
        
        try:
            # Try different methods to control input volume
            if hasattr(self.audio_device, 'setAudioInputVolume'):
                volume = 0 if mute else 100
                self.audio_device.setAudioInputVolume(volume)
                return True
            elif hasattr(self.audio_device, 'setInputVolume'):
                volume = 0 if mute else 100
                self.audio_device.setInputVolume(volume)
                return True
            else:
                # Fallback to input disable/enable if volume control not available
                if mute:
                    return self.disable_robot_microphones()
                else:
                    return self.enable_robot_microphones()
        except Exception as e:
            return False
    
    def is_speaking(self):
        """Check if robot is currently speaking"""
        if not self.connected:
            return False
            
        try:
            # Method 1: Check AnimatedSpeech if available
            if self.animated_speech:
                try:
                    is_running = self.animated_speech.isRunning()
                    if is_running:
                        return True
                except Exception:
                    pass
            
            # Method 2: Check TTS running tasks
            if self.tts:
                try:
                    running_tasks = self.tts.getRunningTasks()
                    if len(running_tasks) > 0:
                        return True
                except Exception:
                    pass
            
            # Method 3: Check our internal speaking flag
            return self.speaking
            
        except Exception:
            return self.speaking
    
    def check_speaking_status(self):
        """Check and return current speaking status"""
        is_currently_speaking = self.is_speaking()
        self.send_response({
            "success": True, 
            "is_speaking": is_currently_speaking,
            "internal_flag": self.speaking
        })
    
    def process_command(self, command):
        """Process incoming command from main script"""
        action = command.get("action", "")
        
        if action == "speak":
            text = command.get("text", "")
            language = command.get("language", "English")
            self.speak(text, language)
            
        elif action == "set_volume":
            volume = command.get("volume", 0.8)
            self.set_volume(volume)
            
        elif action == "set_speed":
            speed = command.get("speed", 1.0)
            self.set_speech_speed(speed)
            
        elif action == "ping":
            self.send_response({"success": True, "message": "pong"})
            
        elif action == "check_speaking":
            self.check_speaking_status()
            
        elif action == "disable_microphones":
            success = self.disable_robot_microphones()
            self.send_response({
                "success": success, 
                "message": "Robot microphones disabled" if success else "Failed to disable microphones",
                "microphones_disabled": self.microphones_disabled
            })
            
        elif action == "enable_microphones":
            success = self.enable_robot_microphones()
            self.send_response({
                "success": success, 
                "message": "Robot microphones enabled" if success else "Failed to enable microphones",
                "microphones_disabled": self.microphones_disabled
            })
            
        elif action == "mute_microphones":
            mute = command.get("mute", True)
            success = self.mute_robot_microphones(mute)
            action_word = "muted" if mute else "unmuted"
            if success:
                message = "Robot microphones " + action_word
            else:
                message = "Failed to " + action_word.replace('ed', 'e') + " microphones"
            self.send_response({
                "success": success, 
                "message": message
            })
            
        # elif action == "test_audio":
        #     self.test_audio_system()
            
        else:
            self.send_response({"success": False, "error": "Unknown action: " + str(action)})
    
    def test_audio_system(self):
        """Test the entire audio system with NAOqi 2.5 specific checks"""
        if not self.connected:
            self.send_response({"success": False, "error": "Not connected to robot"})
            return
        
        try:
            # Test TTS proxy
            volume = self.tts.getVolume()
            language = self.tts.getLanguage()
            voices = self.tts.getAvailableVoices()
            languages = self.tts.getAvailableLanguages()
            
            test_info = {
                "tts_volume": volume,
                "tts_language": language,
                "available_voices": voices,
                "available_languages": languages
            }
            
            # Get current voice
            try:
                current_voice = self.tts.getVoice()
                test_info["current_voice"] = current_voice
            except Exception:
                test_info["current_voice"] = "Unknown"
            
            # Test Audio Device (critical for NAOqi 2.5)
            if self.audio_device:
                try:
                    master_volume = self.audio_device.getOutputVolume()
                    test_info["master_volume"] = master_volume
                    
                    # Check mute status
                    try:
                        is_muted = self.audio_device.isOutputMuted()
                        test_info["audio_muted"] = is_muted
                        
                        if is_muted:
                            self.audio_device.muteAudioOut(False)
                            test_info["unmuted"] = True
                    except Exception as mute_error:
                        test_info["mute_check_error"] = str(mute_error)
                    
                    # NAOqi 2.5 specific: Check if audio output is enabled
                    try:
                        self.audio_device.enableAudioOut(True)
                        test_info["audio_output_enabled"] = True
                    except Exception as enable_error:
                        test_info["audio_enable_error"] = str(enable_error)
                    
                    # Set volume to ensure it's audible for NAOqi 2.5
                    if master_volume < 80:
                        self.audio_device.setOutputVolume(95)
                        test_info["volume_adjusted"] = True
                        test_info["new_master_volume"] = 95
                        
                except Exception as audio_error:
                    test_info["audio_device_error"] = str(audio_error)
            else:
                test_info["audio_device"] = "Not available"
                
            # NAOqi 2.5: Try to get more audio system info
            try:
                # Check if ALAudioRecorder is available (indicates working audio system)
                audio_recorder = ALProxy("ALAudioRecorder", self.robot_ip, self.robot_port)
                test_info["audio_recorder_available"] = True
            except Exception:
                test_info["audio_recorder_available"] = False
            
            # Test speech using multiple methods (NAOqi 2.5 compatibility)
            speech_test_results = []
            
            # Test 1: ALAnimatedSpeech
            if self.animated_speech:
                try:
                    self.animated_speech.say("Audio test one")
                    speech_test_results.append("ALAnimatedSpeech: SUCCESS")
                except Exception as anim_error:
                    speech_test_results.append("ALAnimatedSpeech: FAILED - " + str(anim_error))
            
            # Test 2: Regular TTS say
            try:
                self.tts.say("Audio test two")
                speech_test_results.append("TTS say: SUCCESS")
            except Exception as tts_error:
                speech_test_results.append("TTS say: FAILED - " + str(tts_error))
            
            # Test 3: Async TTS (NAOqi 2.5 preferred method)
            try:
                task_id = self.tts.post.say("Audio test three")
                self.tts.wait(task_id, 10000)
                speech_test_results.append("TTS async: SUCCESS")
            except Exception as async_error:
                speech_test_results.append("TTS async: FAILED - " + str(async_error))
            
            test_info["speech_tests"] = speech_test_results
            
            # Final volume check after tests
            try:
                final_tts_volume = self.tts.getVolume()
                if self.audio_device:
                    final_master_volume = self.audio_device.getOutputVolume()
                    test_info["final_volumes"] = {
                        "tts_volume": final_tts_volume,
                        "master_volume": final_master_volume
                    }
            except Exception:
                pass
            
            self.send_response({"success": True, "message": "Audio system test completed", "details": test_info})
            
        except Exception as e:
            self.send_response({"success": False, "error": "Audio test failed: " + str(e)})


def main():
    """Main entry point"""
    if len(sys.argv) != 3:
        print(json.dumps({"success": False, "error": "Usage: python naoqi_bridge.py <robot_ip> <robot_port>"}))
        sys.exit(1)
    
    robot_ip = sys.argv[1]
    robot_port = sys.argv[2]
    
    # Create bridge instance
    bridge = PepperBridge(robot_ip, robot_port)
    
    # Main command loop
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                command = json.loads(line)
                bridge.process_command(command)
            except ValueError as e:
                bridge.send_response({"success": False, "error": "Invalid JSON: " + str(e)})
            except Exception as e:
                bridge.send_response({"success": False, "error": "Command processing failed: " + str(e)})
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(json.dumps({"success": False, "error": "Bridge error: " + str(e)}))


if __name__ == "__main__":
    main()