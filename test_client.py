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

    # Show available models
    print("\n📋 Available models:")
    models = client.get_models()
    for model in models:
        print(f"  - {model}")

    # Set default model to the first available model (if any)
    if models:
        client.set_default_model(models[0])
        print(f"[TEST] Default model set to: {models[0]}")
    else:
        print("[WARN] No models available to set as default.")

    # Health check
    print("Checking LLM service health...")
    if not client.health_check():
        print("❌ LLM service is not available. Make sure the server is running.")
        print("Start the server with: cd ../baoagent-llm-server && ./scripts/start_server.sh")
        return

    print("✅ LLM service is healthy")

    # Test simple prompt with default model
    print(f"\n🔍 Testing simple prompt (default model: {client.default_model})...")
    response = client.simple_prompt("Hello! You are BAO Agent. Please respond with exactly 'BaoAgent is working!'")
    print(f"Response: {response}")

    # Test simple prompt with per-call model override (if more than one model)
    if len(models) > 1:
        print(f"\n🔍 Testing simple prompt (override model: {models[1]})...")
        response = client.simple_prompt("Hello! Please respond with exactly 'BaoAgent override model!'", model=models[1])
        print(f"Response: {response}")

    # Test chat with default model
    print(f"\n💬 Testing chat (default model: {client.default_model})...")
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "Now what is 4+4?"}
    ]
    response = client.chat(messages, max_tokens=50)
    print(f"Chat response: {response}")

    # Test chat with per-call model override (if more than one model)
    if len(models) > 1:
        print(f"\n💬 Testing chat (override model: {models[2]})...")
        response = client.chat(messages, max_tokens=50, model=models[2])
        print(f"Chat response: {response}")

    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
