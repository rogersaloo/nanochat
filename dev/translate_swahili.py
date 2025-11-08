import os
import requests
import re

from huggingface_hub import login
# login()
from datasets import load_dataset
from datasets import load_from_disk

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SUBSET_NAME = "sample-10BT"
SPLIT = "train"
TEXT_COLUMN = "text"
LOCAL_SAVE_PATH = "fineweb_swhaili_translated"

print(f"Loading dataset { DATASET_NAME}, subset {SUBSET_NAME}, split {SPLIT}")

dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)

print(f"Dataset loaded with {len(dataset)} examples")

API_ENDPOINT = "http://localhost:8001/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "rao2025"
}

def translate_batch_via_api(batch, api_endpoint, headers, text_column):
    translated_texts = []
    batch_number = 0
    if batch_number < 1:
        for text in batch[text_column]:
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate the following text into Swahili Language."},
                {"role": "user", "content": f"Translate this text to Swahili: Only provide the translation, no explanations:'{text}'"}
            ]

            payload = {
                "model": "Deepseek-V3-Terminus",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.1
            }
            try:
                response = requests.post(api_endpoint, headers=headers, json=payload)
                response.raise_for_status()

                result = response.json()
                raw_response_text = result["choices"][0]["message"]["content"]
                clean_text = re.sub(
                    pattern=r'<think>.*?</think>', 
                    repl='', 
                    string=raw_response_text, 
                    flags=re.DOTALL
                )                
                translated_text = clean_text.strip()
            
            except requests.exceptions.RequestException as e:
                print(f"API request failed for text: '{text[:30]}...' Error: {e}")
                translated_text = f"[PARSING FAILED {e}]"
            except (KeyError, IndexError) as e:
                print(f"failed to parse API response for text: '{text[:30]}...' Error: {e}")
                translated_text = f"[PARSING FAILED {e}]"
            batch_number += 1
            print(f"Batch No{batch_number}: {translated_text}")
            translated_texts.append(translated_text)

    batch["text_swahili"] = translated_texts
    return batch


BATCH_SIZE = 8
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
