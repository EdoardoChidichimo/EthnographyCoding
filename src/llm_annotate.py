from openai import OpenAI
import requests
from config import LLM_MODELS
from utils import load_ritual_features, format_prompt, repair_json, validate_and_normalise_output
import threading
import time
from datetime import datetime, timedelta
import random

# Load features once at module level
RITUAL_FEATURES = load_ritual_features()

# Thread-local storage for API clients and rate limiting
thread_local = threading.local()

# Global rate limit tracking
class RateLimiter:
    def __init__(self, tokens_per_min=10000, max_retries=5):
        self.tokens_per_min = tokens_per_min
        self.max_retries = max_retries
        self.lock = threading.Lock()
        self.last_reset = datetime.now()
        self.tokens_used = 0
        
    def wait_if_needed(self, requested_tokens):
        with self.lock:
            now = datetime.now()
            # Reset counter if a minute has passed
            if now - self.last_reset >= timedelta(minutes=1):
                self.tokens_used = 0
                self.last_reset = now
            
            # Calculate wait time if we would exceed the limit
            if self.tokens_used + requested_tokens > self.tokens_per_min:
                wait_time = 60 - (now - self.last_reset).total_seconds()
                return max(0, wait_time)
            
            self.tokens_used += requested_tokens
            return 0

rate_limiter = RateLimiter()

def get_openai_client(model):
    """Get or create thread-local OpenAI client"""
    if not hasattr(thread_local, "openai_client"):
        thread_local.openai_client = OpenAI(api_key=LLM_MODELS[model]["api_key"])
    return thread_local.openai_client

def process_ritual(ritual_number, text, model):
    """Process a single ritual text with specified model."""
    try:
        if model.startswith("gpt"):
            result = query_openai(text, model)
        elif model.startswith("claude"):
            result = query_anthropic(text, model)
        elif model.startswith("mistral"):
            result = query_huggingface(text, model)
        else:
            print(f"Unknown model type: {model}")
            return None

        if result:
            result['ritual_number'] = ritual_number
        return result
    except Exception as e:
        print(f"Error processing ritual {ritual_number} with {model}: {e}")
        return None

def query_openai_with_backoff(client, messages, model, max_retries=5):
    """Query OpenAI API with exponential backoff for rate limits"""
    base_delay = 1
    for attempt in range(max_retries):
        try:
            # Estimate token count (rough approximation)
            estimated_tokens = sum(len(m["content"].split()) * 1.5 for m in messages)
            
            # Check rate limit before making request
            wait_time = rate_limiter.wait_if_needed(estimated_tokens)
            if wait_time > 0:
                time.sleep(wait_time)
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0
            )
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + (random.random() * 0.1)
                    print(f"Rate limit reached, waiting {delay:.2f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                    continue
            raise

def query_openai(text, model):
    """Query OpenAI's GPT models with rate limiting and retries."""
    try:
        client = get_openai_client(model)
        messages = [
            {"role": "system", "content": "You are a coding assistant. Respond with ONLY a JSON object containing the requested feature values. For binary features, use exactly 0 or 1."},
            {"role": "user", "content": format_prompt(RITUAL_FEATURES, text)}
        ]
        
        response = query_openai_with_backoff(client, messages, model)
        if response:
            result = repair_json(response.choices[0].message.content)
            return validate_and_normalise_output(result, RITUAL_FEATURES) if result else None
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def query_anthropic(text, model):
    """Query Claude from Anthropic."""
    try:
        headers = {
            "x-api-key": LLM_MODELS[model]["api_key"],
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": format_prompt(RITUAL_FEATURES, text)}],
            "max_tokens": 1000
        }
        
        with requests.Session() as session:
            response = session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            content = response.json().get("content", [{}])[0].get("text", "")
            result = repair_json(content)
            return validate_and_normalise_output(result, RITUAL_FEATURES) if result else None
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return None

def query_huggingface(text, model):
    """Query models via Hugging Face."""
    try:
        headers = {
            "Authorization": f"Bearer {LLM_MODELS[model]['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": format_prompt(RITUAL_FEATURES, text),
            "parameters": {"max_new_tokens": 1000}
        }
        
        with requests.Session() as session:
            model_path = LLM_MODELS[model].get("model_path", model)
            response = session.post(
                f"https://api-inference.huggingface.co/models/{model_path}",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            content = response.json().get("generated_text", "")
            result = repair_json(content)
            return validate_and_normalise_output(result, RITUAL_FEATURES) if result else None
    except Exception as e:
        print(f"HuggingFace API error: {e}")
        return None