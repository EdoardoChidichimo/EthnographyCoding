import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or ""
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or ""

LLM_MODELS = {
    "o3-mini": {"api_key": OPENAI_API_KEY, "provider": "openai"}, 
    "gpt-4o-mini": {"api_key": OPENAI_API_KEY, "provider": "openai"}, 
    "gpt-3.5-turbo": {"api_key": OPENAI_API_KEY, "provider": "openai"},  
    "gpt-3.5-turbo-16k": {"api_key": OPENAI_API_KEY, "provider": "openai"}, 
    "claude-2": {"api_key": ANTHROPIC_API_KEY, "provider": "anthropic"},
    "mistral-7b": {"api_key": HUGGINGFACE_API_KEY, "provider": "huggingface"},
}