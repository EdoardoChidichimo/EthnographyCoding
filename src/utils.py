import json
from pathlib import Path
import pandas as pd

def load_ritual_features():
    """
    Loads ritual features from ritual_features.csv.
    Expected CSV columns: 'ritual_feature' and 'ritual_description'
    Returns a list of strings formatted as "feature: description".
    """
    current_dir = Path(__file__).parent
    features_path = current_dir.parent / "data" / "ritual_features.csv"
    df = pd.read_csv(features_path)
    features = [f"{row['ritual_feature']}: {row['ritual_description']}" for _, row in df.iterrows()]
    return features

def get_feature_names():
    """
    Loads ritual_features.csv and returns a list of normalised ritual feature names.
    Normalisation: lower-case and underscores instead of spaces.
    """
    current_dir = Path(__file__).parent
    features_path = current_dir.parent / "data" / "ritual_features.csv"
    df = pd.read_csv(features_path)
    names = df["ritual_feature"].tolist()
    # Normalise: lower-case and replace spaces with underscores.
    normalised_names = [name.strip().lower().replace(" ", "_") for name in names]
    return normalised_names

def format_prompt(features, text_section):
    """
    Formats the prompt for LLMs using the ritual features list and the text section.
    
    Parameters:
        features (list): A list of strings, where each string is "feature: description".
        text_section (str): The ethnographic passage text.
    
    Returns:
        str: The formatted prompt.
    """
    feature_list = "\n- " + "\n- ".join(features)
    return f"""
    You are an expert in sociocultural anthropology. Analyse the following ethnographic passage and determine the values for the cultural features as specified below. The output should be in JSON format.

    **Cultural Features & Options:**  
    {feature_list}

    **Text:**  
    "{text_section}"

    **Output format:**  
    {{"synchronised dancing": 1, "ritual_complexity": "moderate", ...}}
    """

def repair_json(text):
    """
    Attempts to repair a JSON string by extracting the substring between the first '{' and the last '}'.
    Returns the loaded JSON or None if it cannot be fixed.
    """
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}") + 1
        try:
            repaired = text[start:end]
            return json.loads(repaired)
        except Exception as e:
            print("Unable to repair JSON:", e)
            return None

def normalise_result(result):
    """
    Normalises the JSON output by standardising keys (e.g., lower-case, underscores)
    and trimming string values.
    """
    normalised = {}
    for key, value in result.items():
        normalised_key = key.strip().lower().replace(" ", "_")
        if isinstance(value, str):
            normalised_value = value.strip().lower()
        else:
            normalised_value = value
        normalised[normalised_key] = normalised_value
    return normalised

def aggregate_results(section_results):
    """
    Aggregates results across sections. For each model, combines all feature values.
    
    Returns:
        dict: A dictionary with models as keys and aggregated features as values.
    """
    aggregated = {}
    for model, results in section_results.items():
        model_agg = {}
        for result in results:
            for key, value in result.items():
                model_agg.setdefault(key, []).append(value)
        aggregated[model] = model_agg
    return aggregated