import pandas as pd
from pathlib import Path
from llm_annotate import process_sections, aggregate_results
from utils import get_feature_names

def main():
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data"
    results_dir = current_dir.parent / "results"

    # Load ritual_texts.csv (expected structure: ritual_number, paragraph)
    ritual_texts_path = data_dir / "ritual_texts.csv"
    ritual_texts_df = pd.read_csv(ritual_texts_path)
    
    # (Since there's only one paragraph per ritual, grouping is optional.)
    grouped = ritual_texts_df.groupby("ritual_number")["paragraph"].apply(lambda x: " ".join(x)).reset_index()

    # Get normalised feature names (to match those in the LLM outputs)
    feature_names = get_feature_names()
    normalised_feature_names = [name.strip().lower().replace(" ", "_") for name in feature_names]

    nested_results = {}   # For the nested structure: ritual_number -> {llm_model: {feature: result}}
    flat_rows = []        # For the flat CSV output (for evaluation)

    for idx, row in grouped.iterrows():
        ritual_number = row["ritual_number"]
        ritual_text = row["paragraph"]
        print(f"Processing ritual {ritual_number}...")
        
        section_results = process_sections([ritual_text])
        if section_results:
            aggregated = aggregate_results(section_results)
            nested_results[ritual_number] = aggregated
            
            # Flatten: for each model, create a row with ritual, model, and each feature.
            for model, features_dict in aggregated.items():
                flat_row = {"ritual": ritual_number, "model": model}
                for feat in normalised_feature_names:
                    # features_dict[feat] is a list with a single value (if present)
                    flat_row[feat] = features_dict.get(feat, [None])[0]
                flat_rows.append(flat_row)
        else:
            print(f"No results generated for ritual {ritual_number}.")

    if flat_rows:
        flat_df = pd.DataFrame(flat_rows)
        output_flat = data_dir / "model_predictions.csv"
        flat_df.to_csv(output_flat, index=False)
        print("Flat predictions saved to", output_flat)
    else:
        print("No flat predictions generated.")

    if nested_results:
        import json
        output_nested = results_dir / "final_coded_ethnography.json"
        with open(output_nested, "w") as f:
            json.dump(nested_results, f, indent=4)
        print("Nested results saved to", output_nested)
    else:
        print("No nested results generated.")

if __name__ == "__main__":
    main()