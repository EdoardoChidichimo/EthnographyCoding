import openai
import json
import time
import requests
import asyncio
from config import LLM_MODELS
from utils import load_ritual_features, format_prompt, repair_json, normalise_result
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RITUAL_FEATURES = load_ritual_features()

def query_openai(text_section, model="gpt-4"):
    """Queries OpenAI's GPT models (GPT-4, GPT-3.5)."""
    api_key = LLM_MODELS[model]["api_key"]
    prompt = format_prompt(RITUAL_FEATURES, text_section)
    try:
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["choices"][0]["message"]["content"]
        result = repair_json(content) or {}
        return normalise_result(result) if result else None
    except openai.error.RateLimitError:
        logger.warning(f"Rate limit exceeded for {model}. Consider increasing backoff.")
        raise
    except openai.error.Timeout:
        logger.warning(f"Request timed out for {model}")
        raise
    except openai.error.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error querying {model}: {e}")
        raise

def query_anthropic(text_section, model="claude-2"):
    """Queries Claude-2 from Anthropic API."""
    api_key = LLM_MODELS[model]["api_key"]
    prompt = format_prompt(RITUAL_FEATURES, text_section)
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200
    }
    
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        content = response.json().get("content", [{}])[0].get("text", "")
        result = repair_json(content) or {}
        return normalise_result(result) if result else None
    except requests.exceptions.Timeout:
        logger.warning(f"Request timed out for {model}")
        raise
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {model}")
        else:
            logger.error(f"HTTP error from Anthropic API: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error querying {model}: {e}")
        raise

def query_huggingface(text_section, model="mistral-7b"):
    """Queries Mistral-7B or LLaMA via Hugging Face API."""
    api_key = LLM_MODELS[model]["api_key"]
    prompt = format_prompt(RITUAL_FEATURES, text_section)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    
    model_path = model  # You might need to use a full path like "mistralai/Mistral-7B-Instruct-v0.1"
    
    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{model_path}", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json().get("generated_text", "")
        result = repair_json(content) or {}
        return normalise_result(result) if result else None
    except requests.exceptions.Timeout:
        logger.warning(f"Request timed out for {model}")
        raise
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {model}")
        else:
            logger.error(f"HTTP error from Hugging Face API: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error querying {model}: {e}")
        raise

# --- Asynchronous Query Wrapper with Retry and Dynamic Rate Limiting ---

async def async_query(model, text_section, max_retries=3):
    """
    Asynchronously query an LLM with a retry mechanism and exponential backoff.
    """
    backoff_time = 2  # Initial backoff time in seconds
    
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
        except (openai.error.RateLimitError, requests.exceptions.HTTPError) as e:
            if attempt < max_retries - 1:
                wait_time = backoff_time * (2 ** attempt)
                logger.warning(f"Rate limit/HTTP error for {model}. Waiting {wait_time}s before retry.")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Max retries exceeded for {model}")
                return model, None
        except (openai.error.Timeout, requests.exceptions.Timeout):
            if attempt < max_retries - 1:
                logger.warning(f"Timeout for {model}. Retrying...")
                await asyncio.sleep(backoff_time)
            else:
                logger.error(f"Max retries exceeded for {model} due to timeouts")
                return model, None
        except Exception as e:
            logger.error(f"Unhandled error for {model}: {e}")
            return model, None
    return model, None

async def process_sections_async(sections):
    """
    Asynchronously process each section (ritual text) through each LLM model.
    """
    tasks = []
    for i, section in enumerate(sections):
        logger.info(f"Processing Section {i+1}/{len(sections)}...")
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