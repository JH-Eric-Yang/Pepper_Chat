#!/usr/bin/env python3
"""
Agent Manager - CLI utility for managing Pepper robot agents
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def save_config(config: Dict[str, Any]):
    """Save configuration to config.json"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Configuration saved successfully.")
    except Exception as e:
        print(f"Error saving config: {e}")
        sys.exit(1)

def list_agents(config: Dict[str, Any]):
    """List all available agents"""
    current_agent = config["agents"]["current_agent"]
    available_agents = config["agents"]["available_agents"]
    
    print("\nAvailable Agents:")
    print("=" * 60)
    for key, agent in available_agents.items():
        marker = " [CURRENT]" if key == current_agent else ""
        print(f"\n{key}{marker}")
        print(f"  Name: {agent['name']}")
        print(f"  Description: {agent['description']}")
        print(f"  System Prompt: {agent['system_prompt'][:100]}...")

def switch_agent(config: Dict[str, Any], agent_key: str):
    """Switch to a different agent"""
    available_agents = config["agents"]["available_agents"]
    
    if agent_key in available_agents:
        config["agents"]["current_agent"] = agent_key
        save_config(config)
        agent_info = available_agents[agent_key]
        print(f"Switched to agent: {agent_info['name']} - {agent_info['description']}")
    else:
        print(f"Agent '{agent_key}' not found.")
        print(f"Available agents: {list(available_agents.keys())}")
        sys.exit(1)

def show_current(config: Dict[str, Any]):
    """Show current agent information"""
    current_agent = config["agents"]["current_agent"]
    available_agents = config["agents"]["available_agents"]
    
    if current_agent in available_agents:
        agent = available_agents[current_agent]
        print(f"\nCurrent Agent: {current_agent}")
        print(f"Name: {agent['name']}")
        print(f"Description: {agent['description']}")
        print(f"System Prompt:\n{agent['system_prompt']}")
    else:
        print(f"Current agent '{current_agent}' not found in available agents")

def interactive_menu(config: Dict[str, Any]):
    """Interactive menu for agent selection"""
    while True:
        print("\n" + "="*50)
        print("Pepper Agent Manager")
        print("="*50)
        print("1. List all agents")
        print("2. Switch agent")
        print("3. Show current agent")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                list_agents(config)
            elif choice == "2":
                list_agents(config)
                agent_keys = list(config["agents"]["available_agents"].keys())
                print(f"\nEnter agent key from the list above:")
                for i, key in enumerate(agent_keys, 1):
                    print(f"{i}. {key}")
                
                try:
                    selection = input("\nEnter agent key or number: ").strip()
                    if selection.isdigit():
                        idx = int(selection) - 1
                        if 0 <= idx < len(agent_keys):
                            selected_key = agent_keys[idx]
                        else:
                            print("Invalid number")
                            continue
                    else:
                        selected_key = selection
                    
                    switch_agent(config, selected_key)
                    config = load_config()  # Reload to get updated config
                except (ValueError, IndexError):
                    print("Invalid selection")
                    
            elif choice == "3":
                show_current(config)
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Pepper Robot Agent Manager")
    parser.add_argument("--list", action="store_true", help="List all available agents")
    parser.add_argument("--switch", metavar="AGENT_KEY", help="Switch to specified agent")
    parser.add_argument("--current", action="store_true", help="Show current agent information")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive menu")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Handle command line arguments
    if args.list:
        list_agents(config)
    elif args.switch:
        switch_agent(config, args.switch)
    elif args.current:
        show_current(config)
    elif args.interactive:
        interactive_menu(config)
    else:
        # Default to interactive mode if no arguments
        interactive_menu(config)

if __name__ == "__main__":
    main()