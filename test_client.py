#!/usr/bin/env python3
"""
Test script for BaoAgent LLM Client
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client import create_baoagent_client

def main():
    print("🧪 Testing BaoAgent LLM Client...")
    
    # Create client
    client = create_baoagent_client()
    
    # Health check
    print("Checking LLM service health...")
    if not client.health_check():
        print("❌ LLM service is not available. Make sure the server is running.")
        print("Start the server with: cd ../baoagent-llm-server && ./scripts/start_server.sh")
        return
    
    print("✅ LLM service is healthy")
    
    # Test simple prompt
    print("\n🔍 Testing simple prompt...")
    response = client.simple_prompt("Hello! Please respond with exactly 'BaoAgent is working!'")
    print(f"Response: {response}")
    
    # Test chat
    print("\n💬 Testing chat...")
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "Now what is 4+4?"}
    ]
    
    response = client.chat(messages, max_tokens=50)
    print(f"Chat response: {response}")
    
    # Test models
    print("\n📋 Available models:")
    models = client.get_models()
    for model in models:
        print(f"  - {model}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
