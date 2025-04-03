import pandas as pd
from config import LLM_MODELS, create_prompt
from utils import load_features
from llm_client import get_llm_client

def process_ethnography(ethnography_number, text, model):
    """Process a single ethnography text with specified model."""
    # Get the LLM client for this model
    client = get_llm_client(model, LLM_MODELS)
    
    features_df = load_features()
    
    result_dict = {'ethnography_number': ethnography_number}
    logprobs_dict = {}  

    for _, row in features_df.iterrows():
        feature_name = row['feature_name']
        feature_description = row['feature_description']
        feature_options = row['feature_options'] if pd.notna(row['feature_options']) else ""
        
        # Create prompt for each feature
        prompt = create_prompt(text, feature_name, feature_description, feature_options)
        
        try:
            # Use the common client interface to generate responses
            response = client.generate(
                prompt=prompt,
                system_prompt="You are an expert in sociocultural anthropology. Respond with ONLY a numerical value for the feature annotation.",
                seed=42,
                get_logprobs=True,
                temperature=0.0,
                max_tokens=10
            )
            
            # Extract response text and logprobs
            result = response["text"].strip()
            
            # Store logprobs if available
            if "logprobs" in response:
                logprobs_dict[feature_name] = response["logprobs"]
            
            # Directly parse the result as a numerical value
            result_dict[feature_name] = float(result) if result.isdigit() else None
                
        except Exception as e:
            print(f"Error processing feature {feature_name} for ethnography {ethnography_number}: {e}")
            result_dict[feature_name] = None

    result_dict['logprobs'] = logprobs_dict
    
    return result_dict