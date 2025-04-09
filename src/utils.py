import json
import pickle
import math
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional


def load_features(feature_file: str = "features.csv") -> pd.DataFrame:
    """
    Load feature descriptions and options from CSV.
    
    Args:
        feature_file: Name of the feature file in the data directory
        
    Returns:
        DataFrame containing feature information
    """
    features_path = Path(__file__).parent.parent / "data" / feature_file
    return pd.read_csv(features_path)


def setup_directories() -> tuple:
    """
    Set up project directories and return their paths.
    
    Returns:
        Tuple of (data_dir, results_dir)
    """
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, results_dir


def process_response_logprobs(logprob_data: Any) -> Dict:
    """
    Convert OpenAI logprobs object to a serializable dictionary.
    Also converts logprobs to probability values for easier interpretation.
    
    Args:
        logprob_data: Logprobs object from OpenAI response
        
    Returns:
        Serializable dictionary with probability information
    """
    if not hasattr(logprob_data, 'content'):
        return {'content': []}
    
    serializable_logprobs = {
        'content': [
            {
                'token': item.token,
                'probability': math.exp(item.logprob),
                'top_logprobs': [
                    {'token': top.token, 'probability': math.exp(top.logprob)} 
                    for top in (item.top_logprobs or [])
                ] if hasattr(item, 'top_logprobs') and item.top_logprobs else []
            }
            for item in logprob_data.content
        ]
    }
    
    return serializable_logprobs


def save_results(results_list: List[Dict], 
                 feature_names: List[str], 
                 output_file: Path,
                 logprobs_data: Optional[Dict] = None,
                 reasoning_data: Optional[Dict] = None) -> None:
    """
    Save processing results to CSV and JSON. All data (results, logprobs, reasoning) is saved in a single JSON file.
    
    Args:
        results_list: List of dictionaries containing annotation results
        feature_names: List of feature names to ensure consistent column order
        output_file: Path to save the CSV results
        logprobs_data: Optional dictionary of logprobs data
        reasoning_data: Optional dictionary of chain of thought reasoning data
    """
    # Convert results to DataFrame and ensure column order
    results_df = pd.DataFrame(results_list)
    
    # Ensure all feature columns exist
    for feature in feature_names:
        if feature not in results_df.columns:
            results_df[feature] = None
    
    # Reorder columns to match feature order
    results_df = results_df[feature_names]
    
    # Save annotations to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    
    # Create a structured JSON output
    json_output = {
        "annotations": results_df.to_dict(orient='records'),
        "logprobs": logprobs_data if logprobs_data else {},
        "reasoning": reasoning_data if reasoning_data else {}
    }
    
    # Save all data to a single JSON file
    json_file = output_file.with_suffix('.json')
    try:
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved complete results to {json_file}")
    except TypeError as e:
        # Fall back to pickle if JSON serialization fails
        print(f"Warning: Error serializing data to JSON: {e}")
        pickle_file = json_file.with_suffix('.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(json_output, f)
        print(f"Saved complete results to {pickle_file} (using pickle format)")
    
            
    return results_df


def retry_with_backoff(func, max_retries=3, initial_wait=20):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_wait: Initial wait time in seconds
        
    Returns:
        Result of the function call
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = initial_wait * (attempt + 1)
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise 