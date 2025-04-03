import json
from openai import OpenAI
import pandas as pd
from pathlib import Path
from config import create_prompt, LLM_MODELS

client = OpenAI(api_key=LLM_MODELS['gpt-4o-mini']['api_key'])

ethnography_texts_df = pd.read_csv(Path(__file__).parent.parent / "data" / "ethnography_texts.csv")
features_df = pd.read_csv(Path(__file__).parent.parent / "data" / "features.csv")

MAX_REQUESTS_PER_BATCH = 50000

batch_tasks = []

for _, ethnography_row in ethnography_texts_df.iterrows():
    ethnography_number = ethnography_row['ethnography_number']
    ethnography_text = ethnography_row['paragraph']
    
    for feature_idx, feature_row in features_df.iterrows():
        feature_name = feature_row['feature_name']
        feature_description = feature_row['feature_description']
        feature_options = feature_row['feature_options'] if pd.notna(feature_row['feature_options']) else ""
        
        # Create prompt for each feature
        prompt = create_prompt(ethnography_text, feature_name, feature_description, feature_options)
        
        # Create task for batch processing
        task = {
            "custom_id": f"ethnography-{ethnography_number}-feature-{feature_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are an expert in sociocultural anthropology. Respond with ONLY a numerical value for the feature annotation."},
                    {"role": "user", "content": prompt}
                ],
                "seed": 42, 
                "max_tokens": 3
            }
        }
        batch_tasks.append(task)

# Split tasks into multiple batches if necessary
batch_count = 0
for i in range(0, len(batch_tasks), MAX_REQUESTS_PER_BATCH):
    batch_count += 1
    batch_slice = batch_tasks[i:i + MAX_REQUESTS_PER_BATCH]
    batch_file_path = Path(__file__).parent.parent / "data" / f"batch_tasks_{batch_count}.jsonl"
    with open(batch_file_path, 'w') as file:
        for task in batch_slice:
            file.write(json.dumps(task) + '\n')
    print(f"Batch file created: {batch_file_path}")





# Create and execute batch jobs for each batch file
batch_files = list(Path(__file__).parent.parent.glob("data/batch_tasks_*.jsonl"))

for batch_file_path in batch_files:
    print(f"Uploading batch file: {batch_file_path}")
    batch_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )

    # Create the batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print(f"Batch job created with ID: {batch_job.id}")

    # Retrieve results once the batch job is done
    # Note: This part of the code should be run after the batch job is completed
    # result_file_id = batch_job.output_file_id
    # result = client.files.content(result_file_id).content
    # result_file_name = f"data/batch_job_results_{batch_file_path.stem}.jsonl"
    # with open(result_file_name, 'wb') as file:
    #     file.write(result)

    # print(f"Batch processing complete for {batch_file_path}. Results saved.") 