OPENAI_API_KEY = "sk-XXXXXXXX"
ANTHROPIC_API_KEY = "sk-YYYYYYYY"
HUGGINGFACE_API_KEY = "hf-XXXXXXXX"

LLM_MODELS = {
    "gpt-4": {"api_key": OPENAI_API_KEY, "provider": "openai"},
    "claude-2": {"api_key": ANTHROPIC_API_KEY, "provider": "anthropic"},
    "mistral-7b": {"api_key": HUGGINGFACE_API_KEY, "provider": "huggingface"},
}