import pandas as pd
from pathlib import Path
from llm_annotate import process_ritual, load_features
from config import LLM_MODELS
from tqdm import tqdm

def main():
    """Process rituals and save results to CSV files."""
    # Set up paths
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load ritual texts and features
    ritual_texts_df = pd.read_csv(data_dir / "ritual_texts.csv")
    features_df = load_features()
    feature_names = ['ritual_number'] + features_df['feature_name'].tolist()
    
    # Filter out empty paragraphs
    valid_rituals = ritual_texts_df[ritual_texts_df['paragraph'].notna()].copy()
    
    # TEMPORARY: Take only first 3 rituals for testing
    valid_rituals = valid_rituals.head(3)
    print(f"Processing {len(valid_rituals)} rituals for testing")
    print(f"Ritual numbers: {sorted(valid_rituals['ritual_number'].tolist())}")

    # Process one model at a time
    for model in LLM_MODELS:
        print(f"\nProcessing rituals with {model}...")
        output_file = results_dir / f"{model}_annotations.csv"
        
        # Skip if file exists
        if output_file.exists():
            print(f"Results file already exists for {model}, skipping...")
            continue
        
        results = []
        for _, row in tqdm(valid_rituals.iterrows(), total=len(valid_rituals)):
            ritual_number = int(row['ritual_number'])
            ritual_text = row['paragraph']
            
            result = process_ritual(ritual_number, ritual_text, model)
            if result:
                results.append(result)
                print(f"Completed ritual {ritual_number}")
        
        if results:
            # Convert results to DataFrame and ensure column order
            results_df = pd.DataFrame(results)
            
            # Ensure all feature columns exist
            for feature in feature_names:
                if feature not in results_df.columns:
                    results_df[feature] = None
            
            # Reorder columns to match feature order
            results_df = results_df[feature_names]
            
            # Save to CSV
            results_df.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")
            
            # Print first few rows to verify structure
            print("\nFirst rows of output file:")
            print(results_df.head().to_string())
        
        # Wait between models
        if not model == list(LLM_MODELS.keys())[-1]:
            print("Waiting 30 seconds before next model...")
            import time
            time.sleep(30)

    print("\nProcessing complete")

if __name__ == "__main__":
    main()