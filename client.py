"""
BaoAgent LLM Client
Unified client for all BaoAgent workflows
"""

import openai
import os
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

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
                "model": "claude-3-sonnet-20240229"
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
    
    def _create_client(self) -> openai.OpenAI:
        """Create OpenAI client based on configuration"""
        provider = self.config["provider"]
        provider_config = self.config[provider]
        
        return openai.OpenAI(
            base_url=provider_config["base_url"],
            api_key=provider_config["api_key"]
        )
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             max_tokens: int = 1000,
             temperature: float = 0.7,
             stream: bool = False) -> str:
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
            model = self.config[provider]["model"]
            
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
    
    def simple_prompt(self, prompt: str, **kwargs) -> str:
        """
        Simple prompt interface
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters for chat()
            
        Returns:
            Response text
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def get_models(self) -> List[str]:
        """Get available models"""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            print(f"❌ Error getting models: {e}")
            return []
    
    def health_check(self) -> bool:
        """Check if the LLM service is healthy"""
        try:
            response = self.simple_prompt("Hello", max_tokens=10)
            return "Error:" not in response
        except:
            return False


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
