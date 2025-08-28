#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for robot microphone control functionality
Tests the new microphone disable/enable features in naoqi_bridge.py
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config.json: {e}")
        return {
            "robot": {"ip": "192.168.1.100", "port": 9559},
            "paths": {"python2_executable": r"E:\Project\Robot\Python27\python.exe"}
        }

def test_microphone_control():
    """Test the microphone control functionality"""
    print("Testing Robot Microphone Control...")
    print("=" * 50)
    
    config = load_config()
    robot_ip = config["robot"]["ip"]
    robot_port = config["robot"]["port"]
    python2_path = config["paths"]["python2_executable"]
    bridge_script = Path(__file__).parent / "naoqi_bridge.py"
    
    print(f"Connecting to robot at {robot_ip}:{robot_port}")
    
    # Start bridge process
    try:
        cmd = [python2_path, str(bridge_script), robot_ip, str(robot_port)]
        process = subprocess.Popen(
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
        
        if process.poll() is not None:
            print("Bridge failed to start!")
            return False
        
        print("Bridge started successfully!")
        
        def send_command(command):
            """Send command and get response"""
            command_json = json.dumps(command) + "\n"
            process.stdin.write(command_json)
            process.stdin.flush()
            
            # Read response
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                response = process.stdout.readline().strip()
                attempts += 1
                
                if not response:
                    continue
                
                if response.startswith('{'):
                    try:
                        return json.loads(response)
                    except json.JSONDecodeError:
                        continue
                else:
                    # Skip NAOqi log lines
                    continue
            
            return {"success": False, "error": "No valid response"}
        
        # Test 1: Check initial connection
        print("\n1. Testing initial connection...")
        response = send_command({"action": "ping"})
        print(f"Ping response: {response}")
        
        # Test 2: Disable microphones
        print("\n2. Testing microphone disable...")
        response = send_command({"action": "disable_microphones"})
        print(f"Disable response: {response}")
        
        # Test 3: Enable microphones
        print("\n3. Testing microphone enable...")
        response = send_command({"action": "enable_microphones"})
        print(f"Enable response: {response}")
        
        # Test 4: Mute microphones
        print("\n4. Testing microphone mute...")
        response = send_command({"action": "mute_microphones", "mute": True})
        print(f"Mute response: {response}")
        
        # Test 5: Unmute microphones
        print("\n5. Testing microphone unmute...")
        response = send_command({"action": "mute_microphones", "mute": False})
        print(f"Unmute response: {response}")
        
        # Test 6: Test speech with automatic microphone control
        print("\n6. Testing speech with automatic microphone control...")
        response = send_command({
            "action": "speak", 
            "text": "Testing microphone control during speech",
            "language": "English"
        })
        print(f"Speech response: {response}")
        
        # Test 7: Final status check
        print("\n7. Final status check...")
        response = send_command({"action": "check_speaking"})
        print(f"Speaking status: {response}")
        
        print("\n" + "=" * 50)
        print("Microphone control test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        # Clean up
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        except:
            pass
    
    return True

if __name__ == "__main__":
    success = test_microphone_control()
    if success:
        print("All tests completed successfully!")
    else:
        print("Some tests failed!")
        sys.exit(1)