import pandas as pd
from pathlib import Path
from llm_annotate import process_ethnography
from config import LLM_MODELS
from tqdm import tqdm
from utils import load_features, setup_directories, save_results

def main():
    # Set up directories
    data_dir, results_dir = setup_directories()

    # Load data and features
    ethnography_texts_df = pd.read_csv(data_dir / "ethnography_texts.csv")
    features_df = load_features()
    feature_names = ['ethnography_number'] + features_df['feature_name'].tolist()
    
    # Filter valid ethnographies
    valid_ethnographies = ethnography_texts_df[ethnography_texts_df['paragraph'].notna()].copy()
    
    # TEMPORARY: Take only first 2 ethnographies for testing
    valid_ethnographies = valid_ethnographies.head(2)
    print(f"Processing {len(valid_ethnographies)} ethnographies for testing")
    print(f"Ethnography numbers: {sorted(valid_ethnographies['ethnography_number'].tolist())}")

    # Process one model at a time
    for model in LLM_MODELS:
        print(f"\nProcessing ethnographies with {model}...")
        output_file = results_dir / f"{model}_annotations.csv"
        logprobs_file = results_dir / f"{model}_logprobs.json"
        
        # Skip if file exists
        if output_file.exists() and logprobs_file.exists():
            print(f"Results files already exist for {model}, skipping...")
            continue
        
        results = []
        logprobs = {}

        for _, row in tqdm(valid_ethnographies.iterrows(), total=len(valid_ethnographies)):
            ethnography_number = int(row['ethnography_number'])
            ethnography_text = row['paragraph']
            
            result = process_ethnography(ethnography_number, ethnography_text, model)
            if result:
                if 'logprobs' in result:
                    logprobs[str(ethnography_number)] = result.pop('logprobs')  # Use string key for JSON
                
                results.append(result)
                print(f"Completed ethnography {ethnography_number}")
        
        if results:
            # Save results and logprobs
            df = save_results(
                results_list=results, 
                feature_names=feature_names, 
                output_file=output_file,
                logprobs_data=logprobs,
                logprobs_file=logprobs_file
            )
        
        # Wait between models
        if not model == list(LLM_MODELS.keys())[-1]:
            print("Waiting 30 seconds before next model...")
            import time
            time.sleep(30)

    print("\nProcessing complete")

if __name__ == "__main__":
    main()