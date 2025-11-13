import os
import requests
import re
import time

from huggingface_hub import login
# login()
from datasets import load_dataset
from datasets import load_from_disk
import concurrent.futures


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SUBSET_NAME = "sample-10BT"
SPLIT = "train"
TEXT_COLUMN = "text"
LOCAL_SAVE_PATH = "/raid/.tnp/fineweb_swhaili_translated"
API_ENDPOINT = "http://localhost:8000/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "rao2025"
}
BATCH_SIZE = 256
MAX_RETRIES = 5
BASE_DELAY = 2 # Initial delay in seconds

print(f"Loading dataset { DATASET_NAME}, subset {SUBSET_NAME}, split {SPLIT}")
dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
print(f"Dataset loaded with {len(dataset)} examples")


def translate_single_text(text, api_endpoint, headers):
    
    messages = [
        {"role": "user", "content": f"Translate this text to Swahili: Only provide the translation, no explanations:'{text}'"}
    ]

    payload = {
        "model": "DeepSeek-V3-Terminus",
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    translated_text = f"[FAILED: Max retries exhausted]" # Default failure message
    
    for attempt in range(MAX_RETRIES):
        delay = BASE_DELAY * (2 ** attempt) # Exponential backoff: 2s, 4s, 8s, 16s, 32s...
        
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=90)
            response.raise_for_status() # Catches 4xx/5xx errors
            
            # If successful, parse and break the retry loop
            result = response.json()
            translated_text = result["choices"][0]["message"]["content"].strip()
            
            if attempt > 0:
                print(f"SUCCESS (Attempt {attempt+1}): Translated after retry. Text: '{text[:30]}...'")
            return translated_text # Return the success result
        
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                # Last attempt failed, return the final error message
                print(f"FAILED (Max attempts exhausted): Text: '{text[:30]}...' Error: {e}")
                return f"[PARSING FAILED {e}]"
            else:
                # Log retry and wait using exponential backoff
                print(f"FAILED (Attempt {attempt+1}). Retrying in {delay}s. Text: '{text[:30]}...' Error: {e}")
                time.sleep(delay)
        
        except (KeyError, IndexError) as e:
            # Handle JSON parsing errors - these are usually fatal, no need to retry
            print(f"FAILED (JSON PARSE): Text: '{text[:30]}...' Error: {e}")
            return f"[PARSING FAILED JSON Error: {e}]"
    
    # Fallback return in case retry loop completes without returning
    return translated_text 

def translate_batch_via_api(batch, api_endpoint, headers, text_column):
    texts = batch[text_column]
    
    # Use ThreadPoolExecutor to run requests concurrently
    # max_workers is set to the BATCH_SIZE (8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        
        # Prepare repeated arguments for executor.map
        api_endpoints = [api_endpoint] * len(texts)
        all_headers = [headers] * len(texts)
        
        # executor.map submits all tasks and returns results in the order of submission
        translated_texts = list(executor.map(
            translate_single_text, 
            texts, 
            api_endpoints, 
            all_headers
        ))
        print(f"{translated_texts[:50]}\n---\n")
        
    # The list is now guaranteed to have the same length as the input batch (8)
    batch["text_swahili"] = translated_texts
    return batch



# def translate_batch_via_api(batch, api_endpoint, headers, text_column):
#     translated_texts = []
#     for text in batch[text_column]:
#         messages = [
#             # {"role": "system", "content": "You are a professional translator. Translate the following text into Swahili Language."},
#             {"role": "user", "content": f"Translate this text to Swahili: Only provide the translation, no explanations:'{text}'"}
#         ]

#         payload = {
#             "model": "DeepSeek-V3-Terminus",
#             "messages": messages,
#             "max_tokens": 1024,
#             "temperature": 0.1
#         }
#         try:
#             response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
#             response.raise_for_status()

#             result = response.json()
#             translated_text = result["choices"][0]["message"]["content"].strip()   
#             print(f"{translated_text[:50]}\n---\n")   
        
#         except requests.exceptions.RequestException as e:
#             print(f"API request failed for text: '{text[:30]}...' Error: {e}")
#             translated_text = f"[PARSING FAILED {e}]"
#         except (KeyError, IndexError) as e:
#             print(f"failed to parse API response for text: '{text[:30]}...' Error: {e}")
#             translated_text = f"[PARSING FAILED {e}]"
#         translated_texts.append(translated_text)

#         batch["text_swahili"] = translated_texts
#     return batch


print("Starting translation")
translated_dataset = dataset.map(
    lambda batch: translate_batch_via_api(batch, API_ENDPOINT, HEADERS, TEXT_COLUMN),
    batched = True,
    batch_size = BATCH_SIZE
)
print("Translation complete")

print(f"Saving translated dataset locally to {LOCAL_SAVE_PATH}")
translated_dataset.save_to_disk(LOCAL_SAVE_PATH)
print(f"Successfully saved to the disk at: {LOCAL_SAVE_PATH}")
