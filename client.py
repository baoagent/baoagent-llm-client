"""
BaoAgent LLM Client
Unified client for all BaoAgent workflows
"""

import openai
import os
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()


class BaoAgentLLMClient:
    """
    BaoAgent LLM Client - works with local vLLM or remote APIs
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize BaoAgent LLM client
        
        Args:
            config_path: Path to config file (optional)
        """
        self.config = self._load_config(config_path)
        provider = self.config["provider"]
        # Auto-select available Ollama model if needed
        if provider == "ollama":
            try:
                url = self.config[provider]["base_url"] + "/api/tags"
                resp = requests.get(url)
                data = resp.json()
                available_models = [m["name"] for m in data.get("models", [])]
                configured_model = self.config[provider]["model"]
                if configured_model not in available_models:
                    if available_models:
                        selected_model = available_models[0]
                        print(f"[INFO] Ollama model '{configured_model}' not found. Using available model '{selected_model}' instead.")
                        self.config[provider]["model"] = selected_model
                    else:
                        print("[ERROR] No models available in Ollama server.")
            except Exception as e:
                print(f"[ERROR] Could not fetch Ollama models: {e}")
        # Set global default model
        self.default_model = self.config[provider]["model"]
        self.client = self._create_client()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        
        # Default configuration
        default_config = {
            "provider": "local",
            "local": {
                "base_url": "http://localhost:8000/v1",
                "api_key": "dummy",
                "model": "mistralai/Mistral-7B-Instruct-v0.1"
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-3.5-turbo"
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com/v1",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": "claude-3-opus-20240229",
                "models": [
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-4-opus-20250701",
                    "claude-4-sonnet-20250701"
                ]
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama2"
            }
        }
        
        # Try to load from config file
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                default_config.update(file_config)
        
        # Override with environment variables
        provider = os.getenv("BAOAGENT_LLM_PROVIDER", default_config["provider"])
        default_config["provider"] = provider
        
        return default_config
    
    def _create_client(self) -> Any:
        """Create client based on configuration"""
        provider = self.config["provider"]
        provider_config = self.config[provider]
        if provider in ["ollama", "anthropic"]:
            return None  # No client needed for requests
        else:
            return openai.OpenAI(
                base_url=provider_config["base_url"],
                api_key=provider_config["api_key"]
            )
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             max_tokens: int = 1000,
             temperature: float = 0.7,
             stream: bool = False,
             model: Optional[str] = None) -> str:
        """
        Send chat completion request
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            
        Returns:
            Response content or generator if streaming
        """
        try:
            provider = self.config["provider"]
            model = model or self.default_model
            if provider == "ollama":
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                url = self.config[provider]["base_url"] + "/api/generate"
                resp = requests.post(url, json=payload, stream=True)
                if stream:
                    def response_generator():
                        for line in resp.iter_lines():
                            if line:
                                data = json.loads(line)
                                yield data.get("response", "")
                    return response_generator()
                else:
                    full_response = ""
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            full_response += data.get("response", "")
                            if data.get("done", False):
                                break
                    return full_response
            elif provider == "anthropic":
                url = self.config[provider]["base_url"] + "/messages"
                api_key = self.config[provider]["api_key"]
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                # Anthropic expects only 'user' and 'assistant' roles
                filtered_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages if m["role"] in ("user", "assistant")
                ]
                payload = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": filtered_messages
                }
                resp = requests.post(url, headers=headers, json=payload)
                data = resp.json()
                # Anthropic returns 'content' as a list of dicts with 'text'
                if "content" in data and isinstance(data["content"], list):
                    return "".join([c.get("text", "") for c in data["content"]])
                elif "content" in data:
                    return data["content"]
                else:
                    return data
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream
                )
                
                if stream:
                    return response
                else:
                    return response.choices[0].message.content
                
        except Exception as e:
            print(f"❌ Error calling BaoAgent LLM: {e}")
            return f"Error: {str(e)}"
    
    def simple_prompt(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """
        Simple prompt interface
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters for chat()
            
        Returns:
            Response text
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, model=model, **kwargs)
    
    def get_models(self) -> List[str]:
        """Get available models"""
        try:
            provider = self.config["provider"]
            if provider == "ollama":
                url = self.config[provider]["base_url"] + "/api/tags"
                resp = requests.get(url)
                data = resp.json()
                # Ollama returns a list of models under 'models'
                return [m["name"] for m in data.get("models", [])]
            elif provider == "anthropic":
                return self.config[provider].get("models", [])
            else:
                response = self.client.models.list()
                return [model.id for model in response.data]
        except Exception as e:
            print(f"❌ Error getting models: {e}")
            return []
    
    def health_check(self) -> bool:
        """Check if the LLM service is healthy"""
        try:
            provider = self.config["provider"]
            if provider == "ollama":
                url = self.config[provider]["base_url"] + "/api/tags"
                resp = requests.get(url)
                return resp.status_code == 200
            elif provider == "anthropic":
                # Try a simple prompt
                response = self.simple_prompt("Hello", max_tokens=10)
                return "Error:" not in str(response)
            else:
                response = self.simple_prompt("Hello", max_tokens=10)
                return "Error:" not in response
        except:
            return False

    def set_default_model(self, model_name: str):
        """
        Set the global default model for the current provider.
        """
        provider = self.config["provider"]
        self.default_model = model_name
        self.config[provider]["model"] = model_name
        print(f"[INFO] Default model set to '{model_name}' for provider '{provider}'")


def create_baoagent_client(config_path: Optional[str] = None) -> BaoAgentLLMClient:
    """
    Factory function to create BaoAgent LLM client
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        BaoAgentLLMClient instance
    """
    return BaoAgentLLMClient(config_path)


# Example usage
if __name__ == "__main__":
    print("🤖 BaoAgent LLM Client Test")
    
    client = create_baoagent_client()
    
    # Health check
    if client.health_check():
        print("✅ LLM service is healthy")
    else:
        print("❌ LLM service is not available")
        exit(1)
    
    # Test simple prompt
    response = client.simple_prompt("Hello! I'm testing BaoAgent.")
    print(f"Response: {response}")
    
    # Test chat
    messages = [
        {"role": "user", "content": "What is BaoAgent?"},
        {"role": "assistant", "content": "BaoAgent is an AI startup with multiple workflows."},
        {"role": "user", "content": "That's correct! How are you doing?"}
    ]
    
    response = client.chat(messages)
    print(f"Chat response: {response}")
