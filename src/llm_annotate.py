from openai import OpenAI
from config import LLM_MODELS
import time
import pandas as pd
from pathlib import Path

def load_features():
    """Load feature descriptions and options from CSV"""
    features_path = Path(__file__).parent.parent / "data" / "ritual_features.csv"
    return pd.read_csv(features_path)

def process_ritual(ritual_number, text, model):
    """Process a single ritual text with specified model."""
    client = OpenAI(api_key=LLM_MODELS[model]["api_key"])
    
    # Load features once
    features_df = load_features()
    
    # Create feature description string
    feature_descriptions = []
    for _, row in features_df.iterrows():
        desc = f"{row['feature_name']}: {row['feature_description']}"
        if pd.notna(row['feature_options']):
            desc += f" [Valid values: {row['feature_options']}]"
        feature_descriptions.append(desc)
    
    try:
        messages = [
            {"role": "system", "content": "You are an expert in sociocultural anthropology. Respond with ONLY a JSON object containing feature annotations. For binary features, use exactly 0 or 1. For other features, use only the specified valid values."},
            {"role": "user", "content": f"""Analyse this ritual text and provide annotations for each feature.

            Features to annotate:
            {chr(10).join(feature_descriptions)}

            Text to analyse:
            {text}

            Respond with ONLY a JSON object where keys are feature names and values are your annotations. Use exactly 0 or 1 for binary features."""}
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0
                )
                
                result = response.choices[0].message.content
                # Clean up the result if needed (remove any non-JSON text)
                if "{" in result and "}" in result:
                    result = result[result.find("{"):result.rfind("}") + 1]
                
                import json
                result_dict = json.loads(result)
                result_dict['ritual_number'] = ritual_number
                return result_dict
                
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(20 * (attempt + 1))  # Exponential backoff
                    continue
                raise
                
    except Exception as e:
        print(f"Error processing ritual {ritual_number}: {e}")
        return None