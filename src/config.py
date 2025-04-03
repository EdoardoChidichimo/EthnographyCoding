import os

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or ""
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or ""

# Model configuration dictionary
LLM_MODELS = {
    # OpenAI models
    "gpt-4o-mini": {
        "provider": "openai", 
        "api_key": OPENAI_API_KEY,
        "description": "Cost-effective GPT-4 class model"
    },
    "gpt-4": {
        "provider": "openai", 
        "api_key": OPENAI_API_KEY,
        "description": "Full GPT-4 model"
    },
    "gpt-3.5-turbo": {
        "provider": "openai", 
        "api_key": OPENAI_API_KEY,
        "description": "Balanced performance/cost model"
    },
    
    # Anthropic models  
    "claude-3-opus": {
        "provider": "anthropic", 
        "api_key": ANTHROPIC_API_KEY,
        "description": "Most powerful Claude model"
    },
    "claude-3-sonnet": {
        "provider": "anthropic", 
        "api_key": ANTHROPIC_API_KEY,
        "description": "Balanced Claude model"
    },
    
    # Google/Gemini models
    "gemini-pro": {
        "provider": "google", 
        "api_key": GOOGLE_API_KEY,
        "description": "Google's Gemini Pro model"
    },
    
    # Local Llama models
    "llama-3-8b": {
        "provider": "llama",
        "model_path": "/path/to/llama-model.gguf",  # Update this path for local use
        "description": "Locally running Llama 3 8B model"
    }
}

def create_prompt(ritual_text: str, feature_text: str, feature_description: str, feature_options: str) -> str:
    """
    Create a prompt for a specific feature and ritual.
    
    Args:
        ritual_text: The ritual text to analyze
        feature_text: The name of the feature
        feature_description: Description of the feature
        feature_options: Valid options for the feature
        
    Returns:
        A formatted prompt string
    """
    return f"""
    Analyze the following ritual text for the feature: {feature_text}—{feature_description}
    Respond only with one of the options: {feature_options}
    
    Text: {ritual_text}
    """

# look into batch processing or the user-end interface Mohsen recommended