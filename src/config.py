import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or ""
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or ""

LLM_MODELS = {
    "gpt-4o-mini": {"api_key": OPENAI_API_KEY, "provider": "openai"},  # Faster and cheaper GPT-4
    # "gpt-3.5-turbo": {"api_key": OPENAI_API_KEY, "provider": "openai"},  # Fast backup option
    # "gpt-3.5-turbo-16k": {"api_key": OPENAI_API_KEY, "provider": "openai"},  # For longer rituals
    # Temporarily commented out for GPT comparison
    # "claude-2": {"api_key": ANTHROPIC_API_KEY, "provider": "anthropic"},
    # "mistral-7b": {"api_key": HUGGINGFACE_API_KEY, "provider": "huggingface"},
}