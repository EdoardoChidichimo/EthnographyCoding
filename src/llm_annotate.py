import openai
import json
import time
import requests
import asyncio
from config import LLM_MODELS
from utils import load_ritual_features, format_prompt, repair_json, normalise_result, aggregate_results

RITUAL_FEATURES = load_ritual_features()

def query_openai(text_section, model="gpt-4"):
    """Queries OpenAI's GPT models (GPT-4, GPT-3.5)."""
    api_key = LLM_MODELS[model]["api_key"]
    prompt = format_prompt(RITUAL_FEATURES, text_section)
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
    prompt = format_prompt(RITUAL_FEATURES, text_section)
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
    prompt = format_prompt(RITUAL_FEATURES, text_section)
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

# --- Asynchronous Query Wrapper with Retry and Dynamic Rate Limiting ---

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
    Asynchronously process each section (ritual text) through each LLM model.
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
    return asyncio.run(process_sections_async(sections))