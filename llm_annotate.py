import openai
import json
import time
import requests
import asyncio
from config import LLM_MODELS, CULTURAL_FEATURES, CATEGORICAL_FEATURES

def format_prompt(binary_features, categorical_features, text_section):
    """Formats the prompt for LLMs, including both binary and categorical features."""
    feature_list = "\n- " + "\n- ".join(binary_features)
    categorical_list = "\n".join([f"{key}: {', '.join(values)}" for key, values in categorical_features.items()])

    return f"""
    You are an expert in sociocultural anthropology. Analyse the following ethnographic passage and determine the values for both binary and categorical cultural features. The output should be in JSON format.

    **Binary Features (1 = Present, 0 = Absent):**  
    {feature_list}  

    **Categorical Features & Options:**  
    {categorical_list}

    **Text:**  
    "{text_section}"

    **Output format:**  
    {{"Synchronised dancing": 1, "Ritual complexity": "Moderate", ...}}
    """

def repair_json(text):
    """Attempt to repair JSON strings by extracting the substring between the first '{' and the last '}'."""
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
    Normalise the JSON output by standardising keys (e.g., lower-case, underscores)
    and trimming string values.
    """
    normalized = {}
    for key, value in result.items():
        normalized_key = key.strip().lower().replace(" ", "_")
        if isinstance(value, str):
            normalized_value = value.strip().lower()
        else:
            normalized_value = value
        normalized[normalized_key] = normalized_value
    return normalized

def query_openai(text_section, model="gpt-4"):
    """Queries OpenAI's GPT models (GPT-4, GPT-3.5)."""
    api_key = LLM_MODELS[model]["api_key"]
    prompt = format_prompt(CULTURAL_FEATURES, CATEGORICAL_FEATURES, text_section)
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key
        )
        content = response["choices"][0]["message"]["content"]
        result = repair_json(content) or {}
        return normalise_result(result) if result else None
    except Exception as e:
        print(f"Error querying {model}: {e}")
        return None

def query_anthropic(text_section, model="claude-2"):
    """Queries Claude-2 from Anthropic API."""
    api_key = LLM_MODELS[model]["api_key"]
    prompt = format_prompt(CULTURAL_FEATURES, CATEGORICAL_FEATURES, text_section)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "max_tokens": 200}
    
    try:
        response = requests.post("https://api.anthropic.com/v1/complete", headers=headers, json=payload)
        content = response.json().get("completion", "")
        result = repair_json(content) or {}
        return normalise_result(result) if result else None
    except Exception as e:
        print(f"Error querying {model}: {e}")
        return None

def query_huggingface(text_section, model="mistral-7b"):
    """Queries Mistral-7B or LLaMA via Hugging Face API."""
    api_key = LLM_MODELS[model]["api_key"]
    prompt = format_prompt(CULTURAL_FEATURES, CATEGORICAL_FEATURES, text_section)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    
    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload)
        content = response.json().get("generated_text", "")
        result = repair_json(content) or {}
        return normalise_result(result) if result else None
    except Exception as e:
        print(f"Error querying {model}: {e}")
        return None

# --- Asynchronous query wrapper with retry and dynamic rate limiting ---

async def async_query(model, text_section, max_retries=3):
    """
    Asynchronously query an LLM with a retry mechanism and exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            if model.startswith("gpt"):
                result = await asyncio.to_thread(query_openai, text_section, model)
            elif model.startswith("claude"):
                result = await asyncio.to_thread(query_anthropic, text_section, model)
            else:
                result = await asyncio.to_thread(query_huggingface, text_section, model)
            if result is not None:
                return model, result
        except Exception as e:
            print(f"Error querying {model} on attempt {attempt+1}: {e}")
        # Exponential backoff before retrying
        await asyncio.sleep(2 ** attempt)
    return model, None

async def process_sections_async(sections):
    """
    Asynchronously process each section through each LLM model.
    """
    tasks = []
    for i, section in enumerate(sections):
        print(f"Processing Section {i+1}/{len(sections)}...")
        for model in LLM_MODELS.keys():
            tasks.append(async_query(model, section))
    responses = await asyncio.gather(*tasks)
    results = {model: [] for model in LLM_MODELS.keys()}
    for model, result in responses:
        if result:
            results[model].append(result)
    return results

def process_sections(sections):
    """
    Synchronous wrapper for the asynchronous processing of sections.
    """
    return asyncio.run(process_sections_async(sections))

def aggregate_results(section_results):
    """
    Aggregates results across sections. Customize this to
    reconcile differences in prompt sensitivity and output style.
    For instance, you might average numeric features or use majority vote for categorical ones.
    """
    # Example: simply merge all results per model into a single dict.
    aggregated = {}
    for model, results in section_results.items():
        model_agg = {}
        for result in results:
            for key, value in result.items():
                # Basic normalization: count occurrences (for categorical features)
                model_agg.setdefault(key, []).append(value)
        # Here you can add logic to reconcile differences (e.g., majority vote)
        aggregated[model] = model_agg
    return aggregated