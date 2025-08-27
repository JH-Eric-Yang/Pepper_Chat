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
        self.connected = False
        
        # Connect to robot
        self.connect()
    
    def connect(self):
        """Connect to Pepper robot"""
        try:
            # Temporarily redirect stdout to stderr during NAOqi initialization
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            try:
                # Initialize text-to-speech proxy
                self.tts = ALProxy("ALTextToSpeech", self.robot_ip, self.robot_port)
                
                # Test connection and set default volume
                self.tts.setVolume(0.8)
            finally:
                # Restore original stdout
                sys.stdout = original_stdout
            
            self.connected = True
            self.send_response({"success": True, "message": "Connected to Pepper robot"})
            
        except Exception as e:
            # Make sure stdout is restored even on error
            sys.stdout = original_stdout
            self.connected = False
            self.send_response({"success": False, "error": str(e)})
    
    def send_response(self, response):
        """Send JSON response to main script"""
        try:
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            # Fallback error response
            print(json.dumps({"success": False, "error": "Response encoding failed"}))
            sys.stdout.flush()
    
    def speak(self, text, language="English"):
        """Make robot speak the given text"""
        if not self.connected:
            self.send_response({"success": False, "error": "Not connected to robot"})
            return
        
        try:
            # Temporarily redirect stdout to stderr during NAOqi operations
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            try:
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
                
                # Speak the text
                print()
                self.tts.say(text)
            finally:
                # Restore original stdout
                sys.stdout = original_stdout
            
            self.send_response({"success": True, "message": "Speech completed"})
            
        except Exception as e:
            # Make sure stdout is restored even on error
            sys.stdout = original_stdout
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
            
        else:
            self.send_response({"success": False, "error": "Unknown action: " + str(action)})


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