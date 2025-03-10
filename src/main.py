import pandas as pd
from pathlib import Path
from llm_annotate import process_ritual
from config import LLM_MODELS
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os

# Create a thread-safe lock for file writing
file_lock = threading.Lock()

def ensure_model_file(results_dir, model):
    """Ensure model output file exists with correct structure"""
    model_file = results_dir / f"{model}_predictions.csv"
    if not model_file.exists():
        # Create DataFrame with ritual_number as first column
        df = pd.DataFrame(columns=['ritual_number'])
        df.to_csv(model_file, index=False)
        # Force write to disk
        with open(model_file, 'a') as f:
            f.flush()
            os.fsync(f.fileno())
    return model_file

def process_and_save(row, model, results_dir):
    """Process a single ritual and save results immediately"""
    ritual_number = int(row['ritual_number'])
    ritual_text = row['paragraph']
    
    # Get model output file
    model_file = results_dir / f"{model}_predictions.csv"
    
    # Check if this ritual has already been processed
    if model_file.exists():
        existing_df = pd.read_csv(model_file)
        if ritual_number in existing_df['ritual_number'].values:
            return None
    
    result = process_ritual(ritual_number, ritual_text, model)
    if result:
        # Use lock when writing to ensure thread safety
        with file_lock:
            # Convert result to DataFrame
            result_df = pd.DataFrame([result])
            
            # Ensure ritual_number is the first column
            cols = ['ritual_number'] + [col for col in result_df.columns if col != 'ritual_number']
            result_df = result_df[cols]
            
            if model_file.exists():
                # Read existing file
                existing_df = pd.read_csv(model_file)
                
                # Append new result
                updated_df = pd.concat([existing_df, result_df], ignore_index=True)
                
                # Sort by ritual_number
                updated_df = updated_df.sort_values('ritual_number')
                
                # Save back to file
                updated_df.to_csv(model_file, index=False)
            else:
                # Create new file
                result_df.to_csv(model_file, index=False)
            
            # Force write to disk
            with open(model_file, 'a') as f:
                f.flush()
                os.fsync(f.fileno())
            
            # Also save individual JSON for debugging
            json_file = results_dir / 'json' / f"ritual_{ritual_number}_{model}.json"
            json_file.parent.mkdir(exist_ok=True)
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        
        return result
    return None

def main():
    # Set up paths
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'json').mkdir(exist_ok=True)  # Create subdirectory for JSON files

    # Load ritual texts
    ritual_texts_df = pd.read_csv(data_dir / "ritual_texts.csv")
    
    # Filter out empty paragraphs efficiently
    valid_mask = ritual_texts_df['paragraph'].notna() & (ritual_texts_df['paragraph'].str.strip() != '')
    valid_rituals = ritual_texts_df[valid_mask].copy()
    skipped_rituals = ritual_texts_df[~valid_mask]['ritual_number'].tolist()
    
    # Print summary
    print(f"Found {len(ritual_texts_df)} total rituals")
    print(f"Skipping {len(skipped_rituals)} rituals with no text: {sorted(skipped_rituals)}")
    print(f"Processing {len(valid_rituals)} rituals with text...")

    # Process one model at a time to better manage rate limits
    for model in LLM_MODELS:
        print(f"\nProcessing rituals with {model}...")
        
        # Ensure model output file exists
        model_file = ensure_model_file(results_dir, model)
        
        # Create tasks for this model
        tasks = []
        existing_rituals = set()
        
        # Check existing results
        if model_file.exists():
            existing_df = pd.read_csv(model_file)
            existing_rituals = set(existing_df['ritual_number'].values)
        
        # Only process rituals that haven't been done yet
        for _, row in valid_rituals.iterrows():
            ritual_number = int(row['ritual_number'])
            if ritual_number not in existing_rituals:
                tasks.append(row)
        
        if not tasks:
            print(f"All rituals already processed for {model}, skipping...")
            continue
            
        print(f"Found {len(tasks)} rituals to process...")
        
        # Use fewer workers for rate-limited APIs
        max_workers = 3 if model.startswith("gpt") else 5
        
        # Process rituals with limited concurrency
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for this model
            future_to_task = {
                executor.submit(process_and_save, row, model, results_dir): row['ritual_number']
                for row in tasks
            }
            
            # Process results as they complete
            completed = 0
            with tqdm(total=len(tasks), desc=f"Processing {model}") as pbar:
                for future in as_completed(future_to_task):
                    ritual_number = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            completed += 1
                            print(f"\nCompleted ritual {ritual_number}")
                    except Exception as e:
                        print(f"\nError processing ritual {ritual_number}: {e}")
                    pbar.update(1)
                    
            print(f"Completed {completed}/{len(tasks)} rituals for {model}")
            
            # Verify and sort the final CSV
            if model_file.exists():
                df = pd.read_csv(model_file)
                df = df.sort_values('ritual_number')
                df.to_csv(model_file, index=False)
            
            # Small delay between models to ensure clean rate limit reset
            if not model == list(LLM_MODELS.keys())[-1]:
                print("Waiting 60 seconds before processing next model...")
                time.sleep(60)

    print(f"\nProcessing complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()