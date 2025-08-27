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

# Add NAOqi SDK to Python path
sys.path.append(r"E:\Project\Robot\naoqi_sdk\lib")

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
            
            # Set TTS volume
            self.tts.setVolume(0.8)
            
            # Set master volume using ALAudioDevice (more reliable)
            if self.audio_device:
                self.audio_device.setOutputVolume(80)  # 0-100 range
                
                # Try to check if audio is muted (method may not exist in all versions)
                try:
                    if self.audio_device.isOutputMuted():
                        self.audio_device.muteAudioOut(False)
                except Exception as mute_error:
                    pass  # Method may not exist in this NAOqi version
            
            # For NAOqi 2.5, ensure TTS is properly initialized
            available_voices = self.tts.getAvailableVoices()
            
            # Set default voice if available
            if available_voices:
                self.tts.setVoice(available_voices[0])
            
            self.connected = True
            self.send_response({"success": True, "message": "Connected to Pepper robot"})
            
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
        """Clean text for speech by removing or replacing problematic Unicode characters"""
        if isinstance(text, unicode):
            # Convert Unicode to ASCII-compatible string
            # Replace common Unicode characters with ASCII equivalents
            text = text.replace(u'\u2019', "'")  # Right single quotation mark
            text = text.replace(u'\u2018', "'")  # Left single quotation mark
            text = text.replace(u'\u201c', '"')  # Left double quotation mark
            text = text.replace(u'\u201d', '"')  # Right double quotation mark
            text = text.replace(u'\u2013', '-')  # En dash
            text = text.replace(u'\u2014', '-')  # Em dash
            text = text.replace(u'\u2026', '...')  # Horizontal ellipsis
            
            # Remove emojis and other symbols
            text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Emoticons
            text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # Symbols & pictographs
            text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # Transport & map symbols
            text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # Flags
            text = re.sub(r'[\U00002600-\U000027BF]', '', text)  # Misc symbols
            text = re.sub(r'[\U0001f926-\U0001f937]', '', text)  # Additional emojis
            text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # Other high Unicode
            
            # Normalize Unicode and encode to ASCII
            text = unicodedata.normalize('NFKD', text)
            text = text.encode('ascii', 'ignore').decode('ascii')
        elif isinstance(text, str):
            # Already a byte string, decode and clean
            try:
                text = text.decode('utf-8')
                return self.clean_text_for_speech(text)
            except UnicodeDecodeError:
                # If it fails, assume it's already ASCII
                text = re.sub(r'[^\x00-\x7F]', '', text)  # Remove non-ASCII
        
        # Clean up any remaining issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip()
        
        return text
    
    def speak(self, text, language="English"):
        """Make robot speak the given text"""
        if not self.connected:
            self.send_response({"success": False, "error": "Not connected to robot"})
            return
        
        try:
            # Clean text for speech
            clean_text = self.clean_text_for_speech(text)
            # Check audio volume first
            current_volume = self.tts.getVolume()
            print("current volume": current_volume)
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
            
            # Speak the text - prefer ALAnimatedSpeech when available
            if self.animated_speech:
                try:
                    # Use ALAnimatedSpeech for better speech quality
                    # Don't use speech modifiers for now to test basic functionality
                    self.animated_speech.say(clean_text)
                except Exception as anim_error:
                    # Fallback to regular TTS
                    try:
                        self.tts.say(clean_text)
                    except Exception as tts_error:
                        # Last resort: try post method
                        task_id = self.tts.post.say(clean_text)
                        self.tts.wait(task_id, 10000)
            else:
                # Use regular TTS
                try:
                    self.tts.say(clean_text)
                except Exception as tts_error:
                    # Fallback to post method
                    task_id = self.tts.post.say(clean_text)
                    self.tts.wait(task_id, 10000)
            
            self.send_response({"success": True, "message": "Speech completed", "volume": current_volume, "language": current_language})
            
        except Exception as e:
            self.send_response({"success": False, "error": str(e)})
    
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
            
        elif action == "test_audio":
            self.test_audio_system()
            
        else:
            self.send_response({"success": False, "error": "Unknown action: " + str(action)})
    
    def test_audio_system(self):
        """Test the entire audio system"""
        if not self.connected:
            self.send_response({"success": False, "error": "Not connected to robot"})
            return
        
        try:
            # Test TTS proxy
            volume = self.tts.getVolume()
            language = self.tts.getLanguage()
            voices = self.tts.getAvailableVoices()
            
            test_info = {
                "tts_volume": volume,
                "tts_language": language,
                "available_voices": voices
            }
            
            # Test Audio Device
            if self.audio_device:
                try:
                    master_volume = self.audio_device.getOutputVolume()
                    test_info["master_volume"] = master_volume
                    
                    # Try to check mute status (may not be available)
                    try:
                        is_muted = self.audio_device.isOutputMuted()
                        test_info["audio_muted"] = is_muted
                        
                        if is_muted:
                            self.audio_device.muteAudioOut(False)
                            test_info["unmuted"] = True
                    except Exception as mute_error:
                        test_info["mute_check_error"] = str(mute_error)
                        
                    # Set volume to ensure it's audible
                    if master_volume < 50:
                        self.audio_device.setOutputVolume(80)
                        test_info["volume_adjusted"] = True
                        
                except Exception as audio_error:
                    test_info["audio_device_error"] = str(audio_error)
            else:
                test_info["audio_device"] = "Not available"
            
            # Test a simple TTS
            try:
                if self.animated_speech:
                    self.animated_speech.say("Audio test")
                    test_info["speech_test"] = "ALAnimatedSpeech completed"
                else:
                    self.tts.say("Audio test")
                    test_info["speech_test"] = "TTS completed"
            except Exception as speech_error:
                test_info["speech_test_error"] = str(speech_error)
            
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