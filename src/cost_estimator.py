import pandas as pd
from pathlib import Path
from config import create_prompt

# Constants for token pricing per million tokens
PRICING = {
    "openai-o1": {"input": 15/1_000_000, "output": 60/1_000_000},
    "openai-o1-mini": {"input": 1.10/1_000_000, "output": 4.4/1_000_000}, 
    "openai-o3-mini": {"input": 1.10/1_000_000, "output": 4.4/1_000_000},
    "gpt-4.5-preview": {"input": 75/1_000_000, "output": 150/1_000_000},
    "gpt-4o": {"input": 2.5/1_000_000, "output": 10/1_000_000},
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output": 0.6/1_000_000},
    "gemini2flash": {"input": 0.1/1_000_000, "output": 0.4/1_000_000},
    "gemini2flash-lite": {"input": 0.075/1_000_000, "output": 0.3/1_000_000},
    "llama3-8b": {"input": 0.0004/1_000, "output": 0.4/1_000},
    "llama3.2-3b": {"input": 0.0004/1_000, "output": 0.4/1_000},
    "llama3-70b": {"input": 0.0028/1_000, "output": 2.8/1_000},
    "llama3.1-8b": {"input": 0.0004/1_000, "output": 0.4/1_000},
    "llama3.2-1b": {"input": 0.0004/1_000, "output": 0.4/1_000},
    "claude3.7-sonnet": {"input": 3/1_000_000, "output": 15/1_000_000},
    "claude3.5-sonnet": {"input": 3/1_000_000, "output": 15/1_000_000},
    "claude3.5-haiku": {"input": 0.8/1_000_000, "output": 4/1_000_000},
    "claude3-opus": {"input": 15/1_000_000, "output": 75/1_000_000},
}

def estimate_cost(ethnographic_texts, features_df):
    """
    Estimate the cost of running the experiment based on token usage.
    
    Parameters:
    - ethnographic_texts: List of ethnography texts to process
    - features_df: DataFrame containing feature descriptions
    
    Returns:
    - Dictionary containing token counts and costs for each model

    NOTE: Using batch processing, the cost reduces significantly.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Calculate total tokens
    for text in ethnographic_texts:
        for _, row in features_df.iterrows():
            feature_name = row['feature_name']
            feature_description = row['feature_description']
            feature_options = row['feature_options'] if pd.notna(row['feature_options']) else ""
            
            # Create prompt for each feature
            prompt = create_prompt(text, feature_name, feature_description, feature_options)
            
            # Estimate input tokens (assuming 1 token per character for simplicity)
            input_tokens = len(prompt)
            total_input_tokens += input_tokens
            
            # Assume 1 output token per response
            total_output_tokens += 1
    
    # Calculate costs for each model
    costs = {}
    for model, rates in PRICING.items():
        model_cost = (total_input_tokens * rates["input"]) + (total_output_tokens * rates["output"])
        costs[model] = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_cost": model_cost
        }
    
    return costs, total_input_tokens, total_output_tokens

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    ethnographic_texts_df = pd.read_csv(data_dir / "ethnographic_texts.csv")
    ethnographic_texts = ethnographic_texts_df['paragraph'].dropna().tolist()  # Use actual texts
    features_df = pd.read_csv(data_dir / "features.csv")

    print(f"Processing {len(ethnographic_texts)} ethnographys")
    print(f"Processing {len(features_df)} features")
    
    costs, total_input_tokens, total_output_tokens = estimate_cost(ethnographic_texts, features_df)

    print(f"\nTotal input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"\nEstimated total cost per model:")
    
    for model, results in sorted(costs.items(), key=lambda x: x[1]['total_cost']):
        print(f"{model}: ${results['total_cost']:.4f}")