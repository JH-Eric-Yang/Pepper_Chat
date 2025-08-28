#!/usr/bin/env python3
"""
Robot Microphone Control Script
Simple script to turn robot microphones on/off remotely
"""

import os
import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

def load_config() -> Dict[str, Any]:
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
            "paths": {
                "python2_executable": r"E:\Project\Robot\Python27\python.exe"
            }
        }

class MicrophoneController:
    """Controls robot microphones via NAOqi bridge"""
    
    def __init__(self, robot_ip: str = None, robot_port: int = None, python2_path: str = None):
        config = load_config()
        
        self.robot_ip = robot_ip or config["robot"]["ip"]
        self.robot_port = robot_port or config["robot"]["port"]
        self.python2_path = python2_path or config["paths"]["python2_executable"]
        self.bridge_script = Path(__file__).parent / "naoqi_bridge.py"
        self.process = None
        
    def start_bridge(self) -> bool:
        """Start the Python 2 NAOqi bridge subprocess"""
        try:
            cmd = [
                self.python2_path,
                str(self.bridge_script),
                self.robot_ip,
                str(self.robot_port)
            ]
            
            print(f"Starting NAOqi bridge: {self.robot_ip}:{self.robot_port}")
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for bridge to initialize
            time.sleep(3)
            
            if self.process.poll() is None:
                print("NAOqi bridge started successfully")
                return True
            else:
                print("Bridge failed to start")
                return False
                
        except Exception as e:
            print(f"Failed to start NAOqi bridge: {e}")
            return False
    
    def send_command(self, command: Dict[str, Any]) -> Optional[Dict]:
        """Send command to NAOqi bridge and get response"""
        if not self.process or self.process.poll() is not None:
            print("NAOqi bridge is not running")
            return None
        
        try:
            command_json = json.dumps(command) + "\n"
            self.process.stdin.write(command_json)
            self.process.stdin.flush()
            
            # Read response, skipping any log lines that don't start with '{'
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                response = self.process.stdout.readline().strip()
                attempts += 1
                
                if not response:
                    print("Empty response from bridge")
                    return {"success": False, "error": "Empty response from bridge"}
                
                # Skip NAOqi log lines (they don't start with '{')
                if response.startswith('{'):
                    try:
                        return json.loads(response)
                    except json.JSONDecodeError as json_error:
                        print(f"Invalid JSON response from bridge: '{response}' - {json_error}")
                        return {"success": False, "error": f"Invalid JSON response: {response}"}
                else:
                    # This is a log line, skip it and try the next line
                    continue
            
            print(f"No valid JSON response found after {max_attempts} attempts")
            return {"success": False, "error": "No valid JSON response found"}
            
        except Exception as e:
            print(f"Communication error with bridge: {e}")
            return None
    
    def disable_microphones(self) -> bool:
        """Disable robot's microphones"""
        print("Disabling robot microphones...")
        command = {"action": "disable_microphones"}
        response = self.send_command(command)
        
        if response and response.get("success", False):
            print("✓ Robot microphones disabled successfully")
            return True
        else:
            error_msg = response.get("error", "Unknown error") if response else "No response from bridge"
            print(f"✗ Failed to disable robot microphones: {error_msg}")
            return False
    
    def enable_microphones(self) -> bool:
        """Enable robot's microphones"""
        print("Enabling robot microphones...")
        command = {"action": "enable_microphones"}
        response = self.send_command(command)
        
        if response and response.get("success", False):
            print("✓ Robot microphones enabled successfully")
            return True
        else:
            error_msg = response.get("error", "Unknown error") if response else "No response from bridge"
            print(f"✗ Failed to enable robot microphones: {error_msg}")
            return False
    
    def mute_microphones(self, mute: bool = True) -> bool:
        """Mute/unmute robot's microphones"""
        action_word = "Muting" if mute else "Unmuting"
        print(f"{action_word} robot microphones...")
        
        command = {"action": "mute_microphones", "mute": mute}
        response = self.send_command(command)
        
        if response and response.get("success", False):
            status = "muted" if mute else "unmuted"
            print(f"✓ Robot microphones {status} successfully")
            return True
        else:
            error_msg = response.get("error", "Unknown error") if response else "No response from bridge"
            status = "mute" if mute else "unmute"
            print(f"✗ Failed to {status} robot microphones: {error_msg}")
            return False
    
    def test_connection(self) -> bool:
        """Test connection to robot"""
        print("Testing connection to robot...")
        command = {"action": "ping"}
        response = self.send_command(command)
        
        if response and response.get("success", False):
            print("✓ Connection to robot successful")
            return True
        else:
            print("✗ Failed to connect to robot")
            return False
    
    def stop_bridge(self):
        """Stop the NAOqi bridge subprocess"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                print("Bridge stopped")
            except subprocess.TimeoutExpired:
                self.process.kill()
                print("Bridge forcefully terminated")
            self.process = None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Control robot microphones")
    parser.add_argument("action", choices=["on", "off", "mute", "unmute", "test"], 
                       help="Action to perform")
    parser.add_argument("--ip", help="Robot IP address")
    parser.add_argument("--port", type=int, help="Robot port")
    parser.add_argument("--python2", help="Path to Python 2 executable")
    
    args = parser.parse_args()
    
    # Create controller
    controller = MicrophoneController(
        robot_ip=args.ip,
        robot_port=args.port,
        python2_path=args.python2
    )
    
    try:
        # Start bridge
        if not controller.start_bridge():
            print("Failed to start bridge. Exiting.")
            return 1
        
        # Test connection first
        if not controller.test_connection():
            print("Connection test failed. Exiting.")
            return 1
        
        # Perform requested action
        success = False
        if args.action == "off":
            success = controller.disable_microphones()
        elif args.action == "on":
            success = controller.enable_microphones()
        elif args.action == "mute":
            success = controller.mute_microphones(True)
        elif args.action == "unmute":
            success = controller.mute_microphones(False)
        elif args.action == "test":
            success = True  # Connection test already passed
            print("✓ Connection test completed successfully")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        controller.stop_bridge()

if __name__ == "__main__":
    exit(main())