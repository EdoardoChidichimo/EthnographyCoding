import json
import math
import requests
from typing import Dict, List, Any, Optional, Union, Callable
from openai import OpenAI
from pathlib import Path
from utils import retry_with_backoff

# Import conditionally to handle missing packages gracefully
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


class LLMClient:
    """
    Unified client for calling different LLM providers with consistent handling of 
    seeds, logprobs, and other model-specific parameters.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize LLM client with model configuration.
        
        Args:
            model_config: Dictionary containing model configuration
            {
                "name": Model name
                "provider": Provider name (openai, anthropic, google, llama)
                "api_key": API key
                "other_params": Any other model-specific parameters
            }
        """
        self.config = model_config
        self.provider = model_config.get("provider", "").lower()
        self.name = model_config.get("name", "")
        self.api_key = model_config.get("api_key", "")
        self.client = self._init_client()
        
    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "openai":
            return OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            return Anthropic(api_key=self.api_key)
        elif self.provider == "google" and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            return genai
        elif self.provider == "llama" and LLAMA_AVAILABLE:
            # Llama needs model path from config
            model_path = self.config.get("model_path", "")
            if not model_path:
                raise ValueError("Model path must be provided for Llama models")
            return Llama(model_path=model_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: str = "",
                 seed: int = 42,
                 get_logprobs: bool = True,
                 temperature: float = 0.0,
                 max_tokens: int = 10) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            seed: Random seed for reproducibility
            get_logprobs: Whether to get logprobs/tokens
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing:
            {
                "text": Generated text
                "logprobs": Logprobs if available
                "finish_reason": Reason for finishing
                "model": Model name
            }
        """
        provider_method = getattr(self, f"_generate_{self.provider}", None)
        if not provider_method:
            raise ValueError(f"Generation not implemented for provider: {self.provider}")
        
        return provider_method(
            prompt=prompt, 
            system_prompt=system_prompt,
            seed=seed,
            get_logprobs=get_logprobs,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _generate_openai(self, 
                         prompt: str, 
                         system_prompt: str,
                         seed: int,
                         get_logprobs: bool,
                         temperature: float,
                         max_tokens: int) -> Dict[str, Any]:
        """OpenAI specific implementation."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        def api_call():
            return self.client.chat.completions.create(
                model=self.name,
                messages=messages,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=get_logprobs,
                top_logprobs=5 if get_logprobs else None
            )
        
        response = retry_with_backoff(api_call)
        
        result = {
            "text": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "model": self.name
        }
        
        # Process logprobs if available
        if get_logprobs and hasattr(response.choices[0], 'logprobs'):
            result["logprobs"] = self._process_openai_logprobs(response.choices[0].logprobs)
            
        return result
    
    def _generate_anthropic(self, 
                           prompt: str, 
                           system_prompt: str,
                           seed: int,
                           get_logprobs: bool,
                           temperature: float,
                           max_tokens: int) -> Dict[str, Any]:
        """Anthropic specific implementation."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
        
        def api_call():
            message = self.client.messages.create(
                model=self.name,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return message
        
        response = retry_with_backoff(api_call)
        
        result = {
            "text": response.content[0].text,
            "finish_reason": response.stop_reason,
            "model": self.name
        }
        
        # Claude doesn't directly support logprobs as of now
        if get_logprobs:
            result["logprobs"] = {"note": "Logprobs not available for Anthropic models"}
            
        return result
    
    def _generate_google(self, 
                         prompt: str, 
                         system_prompt: str,
                         seed: int,
                         get_logprobs: bool,
                         temperature: float,
                         max_tokens: int) -> Dict[str, Any]:
        """Google (Gemini) specific implementation."""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package not installed. Install with 'pip install google-generativeai'")
        
        model = self.client.GenerativeModel(model_name=self.name)
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "candidate_count": 1
        }
        
        # Set safety settings to minimal to match other LLM providers
        safety_settings = [
            {
                "category": cat,
                "threshold": "BLOCK_NONE"
            } 
            for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        
        contents = []
        if system_prompt:
            contents.append({"role": "system", "parts": [system_prompt]})
        contents.append({"role": "user", "parts": [prompt]})
        
        def api_call():
            return model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
        response = retry_with_backoff(api_call)
        
        result = {
            "text": response.text,
            "finish_reason": "stop",  # Gemini doesn't specify this clearly
            "model": self.name
        }
        
        # No direct logprobs support in Gemini API
        if get_logprobs:
            result["logprobs"] = {"note": "Logprobs not available for Google Gemini models"}
            
        return result
    
    def _generate_llama(self, 
                       prompt: str, 
                       system_prompt: str,
                       seed: int,
                       get_logprobs: bool,
                       temperature: float,
                       max_tokens: int) -> Dict[str, Any]:
        """Local Llama specific implementation."""
        if not LLAMA_AVAILABLE:
            raise ImportError("Llama package not installed. Install with 'pip install llama-cpp-python'")
        
        # Check if we're using local Llama or API
        if hasattr(self.client, "__call__"):
            # Local Llama implementation
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
            
            def api_call():
                return self.client(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    echo=False
                )
                
            response = retry_with_backoff(api_call)
            
            result = {
                "text": response["choices"][0]["text"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "model": self.name
            }
            
            # Process logprobs if requested and available
            if get_logprobs and "logprobs" in response["choices"][0]:
                result["logprobs"] = self._process_llama_logprobs(response["choices"][0]["logprobs"])
        else:
            # LLaMA API implementation
            url = f"https://api.llama.sh/{self.name}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
            
            data = json.dumps({
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": 5 if get_logprobs else 0,  # Request log probabilities for top 10 tokens
                "echo": False,
                "seed": seed
            })
            
            def api_call():
                response = requests.post(url, headers=headers, data=data)
                response.raise_for_status()
                return response.json()
            
            response = retry_with_backoff(api_call)
            
            result = {
                "text": response["choices"][0]["text"],
                "finish_reason": response["choices"][0].get("finish_reason", "stop"),
                "model": self.name
            }
            
            # Process logprobs if requested and available
            if get_logprobs and "logprobs" in response["choices"][0]:
                result["logprobs"] = self._process_llama_logprobs(response["choices"][0]["logprobs"])
                
        return result
    
    def _process_openai_logprobs(self, logprob_data):
        """Process OpenAI logprobs into a standardized format."""
        if not hasattr(logprob_data, 'content'):
            return {'content': []}
        
        return {
            'content': [
                {
                    'token': item.token,
                    'probability': math.exp(item.logprob),
                    'top_alternatives': [
                        {'token': top.token, 'probability': math.exp(top.logprob)} 
                        for top in (item.top_logprobs or [])
                    ] if hasattr(item, 'top_logprobs') and item.top_logprobs else []
                }
                for item in logprob_data.content
            ]
        }
    
    def _process_llama_logprobs(self, logprob_data):
        """Process Llama logprobs into a standardized format."""
        result = {'content': []}
        
        if not logprob_data:
            return result
            
        for token, logprob in zip(logprob_data.get("tokens", []), logprob_data.get("token_logprobs", [])):
            token_data = {
                'token': token,
                'probability': math.exp(logprob) if logprob is not None else 0,
                'top_alternatives': []
            }
            
            # Check if top_logprobs is available
            top_tokens = logprob_data.get("top_logprobs", [])
            if top_tokens and len(top_tokens) > 0:
                for top_token, top_logprob in top_tokens.items():
                    token_data['top_alternatives'].append({
                        'token': top_token,
                        'probability': math.exp(top_logprob)
                    })
                    
            result['content'].append(token_data)
            
        return result


def get_llm_client(model_name: str, config_dict: Dict[str, Dict]) -> LLMClient:
    """
    Factory function to get an LLM client for a specific model.
    
    Args:
        model_name: Name of the model to use
        config_dict: Dictionary containing model configurations
        
    Returns:
        LLMClient instance for the specified model
    """
    if model_name not in config_dict:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    model_config = config_dict[model_name]
    model_config["name"] = model_name
    
    return LLMClient(model_config) 