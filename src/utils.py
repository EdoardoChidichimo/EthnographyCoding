import json
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_ritual_features():
    """
    Loads ritual features from the CSV file.
    
    Expected CSV structure:
    feature_name,feature_description,feature_options
    
    Returns:
    --------
    list
        List of dictionaries containing feature information
    """
    try:
        features_path = Path(__file__).parent.parent / "data" / "ritual_features.csv"
        features_df = pd.read_csv(features_path)
        
        # Validate required columns
        required_columns = ['feature_name', 'feature_description', 'feature_options']
        missing_columns = [col for col in required_columns if col not in features_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in ritual_features.csv: {missing_columns}")
        
        # Convert DataFrame to list of dictionaries with normalized keys
        features = []
        for _, row in features_df.iterrows():
            feature = {
                'name': row['feature_name'].strip(),
                'description': row['feature_description'].strip(),
                'options': row['feature_options'].strip() if pd.notna(row['feature_options']) else None
            }
            features.append(feature)
            
        logger.info(f"Loaded {len(features)} ritual features from {features_path}")
        return features
        
    except FileNotFoundError:
        logger.error(f"Could not find ritual_features.csv in {features_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("ritual_features.csv is empty")
        raise
    except Exception as e:
        logger.error(f"Error loading ritual features: {e}")
        raise

def get_feature_names():
    """
    Loads ritual_features.csv and returns a list of normalised ritual feature names.
    Normalisation: lower-case and underscores instead of spaces.
    """
    current_dir = Path(__file__).parent
    features_path = current_dir.parent / "data" / "ritual_features.csv"
    df = pd.read_csv(features_path)
    names = df["feature_name"].tolist()
    # Normalise: lower-case and replace spaces with underscores.
    normalised_names = [name.strip().lower().replace(" ", "_") for name in names]
    return normalised_names

def format_prompt(features, text_section):
    """Format the prompt for the LLM."""
    feature_list = []
    for feature in features:
        feature_str = f"{feature['name']}"
        if feature['options']:
            feature_str += f" (valid values: {feature['options']})"
        feature_list.append(feature_str)
    
    return f"""Analyse this text and provide values for each feature. Use ONLY the specified valid values.

    Features:
    {chr(10).join(feature_list)}

    Text:
    {text_section}

    Respond with ONLY a JSON object containing feature names and values. For binary features (0,1), use exactly 0 or 1."""

def validate_and_normalise_output(result, features):
    """Validate and normalise LLM output."""
    if not result:
        return None
        
    normalized = {}
    feature_specs = {f['name']: f for f in features}
    
    for feature_name, spec in feature_specs.items():
        value = result.get(feature_name)
        if value is None:
            continue
            
        if not spec['options']:
            normalized[feature_name] = value
            continue
            
        valid_options = [opt.strip() for opt in spec['options'].split(',')]
        str_value = str(value).strip()
        
        # Direct match
        if str_value in valid_options:
            normalized[feature_name] = int(str_value) if str_value.isdigit() else str_value
            continue
            
        # Try numeric conversion
        try:
            num_value = int(float(str_value))
            if str(num_value) in valid_options:
                normalized[feature_name] = num_value
        except (ValueError, TypeError):
            pass
            
    return normalized

def repair_json(text):
    """Extract and parse JSON from text."""
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except:
            pass
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
    Aggregates results across sections for each model.
    
    Parameters:
    -----------
    section_results : dict
        Dictionary with model names as keys and lists of feature predictions as values
        Format: {model_name: [{'ritual_number': num, 'feature1': val1, ...}, ...]}
        
    Returns:
    --------
    dict
        Aggregated results in the format:
        {model_name: {
            'ritual_numbers': [num1, num2, ...],
            'features': {
                feature_name: [value1, value2, ...]
            }
        }}
    """
    if not section_results:
        logger.warning("No results to aggregate")
        return {}
        
    aggregated = {}
    
    # Process each model's results
    for model, predictions in section_results.items():
        if not predictions:  # Skip if model has no predictions
            continue
            
        # Initialize model's structure
        model_data = {
            'ritual_numbers': [],
            'features': {}
        }
        
        # Aggregate all predictions for this model
        for prediction in predictions:
            if not prediction:  # Skip empty predictions
                continue
                
            # Store ritual number
            ritual_number = prediction.get('ritual_number')
            if ritual_number is not None:
                model_data['ritual_numbers'].append(ritual_number)
            
            # Store feature predictions
            for key, value in prediction.items():
                if key != 'ritual_number':  # Skip the ritual number field
                    if key not in model_data['features']:
                        model_data['features'][key] = []
                    model_data['features'][key].append(value)
        
        # Only add model if it has features
        if model_data['ritual_numbers']:
            aggregated[model] = model_data
            
    return aggregated