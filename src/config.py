import os

CoT = False

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

MAX_REQUESTS_PER_BATCH = 50000

def create_prompt(ethnographic_text: str, feature_name: str, feature_description: str, feature_options: str) -> str:
    
    CoT_prompt = ""
    if CoT:
        CoT_prompt = "First, explain your reasoning: identify relevant text parts, analyse relation to feature, and determine best option. Then, on the final line, provide ONLY your numerical answer."
    
    return f"""
    Analyse this ethnographic text for feature '{feature_name}'.
    
    Feature description: {feature_description}
    Valid response options: {feature_options}
    {CoT_prompt}
    
    Ethnographic text:
    {ethnographic_text}
    """