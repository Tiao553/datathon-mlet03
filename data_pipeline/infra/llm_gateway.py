import os
import requests
import json
import time
import psutil
from typing import Protocol, Any, Dict, Optional
from openai import OpenAI

# -------------------------------------------------------------------------
# Interface Definition
# -------------------------------------------------------------------------

class LLMProvider(Protocol):
    """
    Protocol defining the standard interface for LLM interaction.
    """
    def generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """
        Generates text based on the prompt.
        
        Args:
            prompt: The input text prompt.
            model_name: Optional override for the model name (provider specific).
            **kwargs: Extra arguments like temperature, max_tokens, etc.
        
        Returns:
            The generated string response.
        """
        ...

# -------------------------------------------------------------------------
# Adapters
# -------------------------------------------------------------------------

class OllamaAdapter:
    """
    Adapter for Local Ollama instance.
    """
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        # Default to environment or hardcoded default
        model = model_name or os.getenv("LLM_MODEL_NAME", "gemma3:1b")
        
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("num_predict", 1536),
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.8),
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=300 # 5 min timeout for slow local inference
            )
            response.raise_for_status()
            result = response.json()
            
            # Observability (Keep existing logs for now)
            exec_time = time.time() - start_time
            print(f"[Ollama] Time: {exec_time:.2f}s | Model: {model}")
            
            return result.get('response', '').strip()
            
        except requests.exceptions.RequestException as e:
            print(f"[Ollama Error] Connection failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")

class DeepSeekAdapter:
    """
    Adapter for DeepSeek API (OpenAI Compatible).
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
             # Fallback or error - but we might lazily init
             pass
        self.client = OpenAI(
            api_key=self.api_key or "sk-placeholder", 
            base_url="https://api.deepseek.com/v1"
        )

    def generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        model = model_name or "deepseek-chat"
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("num_predict", 1536)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[DeepSeek Error] API call failed: {e}")
            raise RuntimeError(f"DeepSeek generation failed: {e}")

# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------

def get_llm_provider() -> LLMProvider:
    """
    Returns the configured LLM Provider based on 'LLM_PROVIDER' env var.
    Defaults to 'ollama'.
    """
    provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider_name == "deepseek":
        return DeepSeekAdapter()
    elif provider_name == "ollama":
        return OllamaAdapter()
    else:
        print(f"Warning: Unknown provider '{provider_name}', defaulting to Ollama.")
        return OllamaAdapter()
