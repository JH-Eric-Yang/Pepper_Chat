#!/usr/bin/env python3
"""
Direct test script for the NAOqi bridge
"""
import json
import subprocess
import time
import threading
import queue

def read_responses(process, response_queue):
    """Read responses from the process in a separate thread"""
    while True:
        try:
            line = process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
# print(f"DEBUG: Raw line: {repr(line)}")  # Debug output
            
            if line.startswith('{'):
                # This is a JSON response
                try:
                    response = json.loads(line)
                    response_queue.put(("response", response))
                except json.JSONDecodeError as e:
                    response_queue.put(("log", f"JSON decode error: {line}"))
            else:
                # This is a log line
                response_queue.put(("log", line))
        except Exception as e:
            response_queue.put(("error", f"Reader error: {str(e)}"))
            break

def test_bridge_tts():
    """Test TTS directly through the bridge"""
    
    # MODIFY THIS STRING TO TEST DIFFERENT MESSAGES
    TEST_MESSAGE = "Test"
    TEST_LANGUAGE = "English"  # Should work - valid options: English, French, Japanese, Chinese, German, Spanish
    
    # Start the bridge
    cmd = [
        r"E:\Project\Robot\Python27\python.exe",
        "naoqi_bridge.py",
        "172.20.10.14",
        "9559"
    ]
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Create response queue and reader thread
    response_queue = queue.Queue()
    reader_thread = threading.Thread(target=read_responses, args=(process, response_queue))
    reader_thread.daemon = True
    reader_thread.start()
    
    # Wait for initialization
    print("Waiting for bridge initialization...")
    time.sleep(3)
    
    try:
        # Wait for initial connection response
        print("Waiting for connection response...")
        timeout_start = time.time()
        while time.time() - timeout_start < 10:
            try:
                msg_type, data = response_queue.get(timeout=1)
                if msg_type == "response":
                    print(f"Connection response: {data}")
                    break
                elif msg_type == "log":
                    print(f"Log: {data}")
            except queue.Empty:
                continue
        
        # Skip audio system test and go directly to your message
        print(f"Speaking your message: '{TEST_MESSAGE}'")
        tts_cmd = {"action": "speak", "text": TEST_MESSAGE, "language": TEST_LANGUAGE}
        process.stdin.write(json.dumps(tts_cmd) + "\n")
        process.stdin.flush()
        
        # Read response with more detailed logging
        timeout_start = time.time()
        while time.time() - timeout_start < 20:
            try:
                msg_type, data = response_queue.get(timeout=1)
                if msg_type == "response":
                    print(f"TTS response: {data}")
                    if data.get("success"):
                        print(f"SUCCESS: Robot should have said '{TEST_MESSAGE}'")
                    else:
                        print(f"ERROR: Speech failed - {data.get('error', 'Unknown error')}")
                    break
                elif msg_type == "log":
                    print(f"Log: {data}")
            except queue.Empty:
                continue
        
        # Send ping to verify bridge is still alive
        print("Sending ping...")
        ping_cmd = {"action": "ping"}
        process.stdin.write(json.dumps(ping_cmd) + "\n")
        process.stdin.flush()
        
        timeout_start = time.time()
        while time.time() - timeout_start < 5:
            try:
                msg_type, data = response_queue.get(timeout=1)
                if msg_type == "response":
                    print(f"Ping response: {data}")
                    break
                elif msg_type == "log":
                    print(f"Log: {data}")
            except queue.Empty:
                continue
        
    except Exception as e:
        print(f"Error during test: {e}")
    
    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Bridge process terminated")

if __name__ == "__main__":
    test_bridge_tts()