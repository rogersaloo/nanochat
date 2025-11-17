import os
import requests
import re
import time
import concurrent.futures
import glob 
import logging

from huggingface_hub import login
# login()
from datasets import load_dataset
from datasets import load_from_disk, concatenate_datasets

DATASET_NAME = os.getenv("DATASET_NAME", "HuggingFaceFW/fineweb-edu")
SUBSET_NAME = os.getenv("SUBSET_NAME", "sample-10BT")
SPLIT = os.getenv("SPLIT", "train")
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "text")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "/raid/.tnp/fineweb_edu_raw")
LOCAL_SAVE_PATH = os.getenv("LOCAL_SAVE_PATH", "/raid/.tnp/fineweb_edu_translated")
FINAL_PARQUET_PATH = os.getenv("FINAL_PARQUET_PATH", "/raid/.tnp/fineweb_edu_final.parquet")

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000/v1/chat/completions")
HEADERS = {
    "Content-Type": "application/json",
}

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
BASE_DELAY = int(os.getenv("BASE_DELAY", "2"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8000"))
PROMPT_TEMPLATE = str(os.getenv("PROMPT_TEMPLATE", "Translate"))
MODEL_ID = str(os.getenv("MODEL_ID", "DeepSeek-V3-Terminus"))

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def translate_single_text(text, api_endpoint, headers):
    messages = [
        {"role": "user", "content": f"{PROMPT_TEMPLATE}'{text}'"}
    ]

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    translated_text = f"[FAILED: Max retries exhausted]"
    
    for attempt in range(MAX_RETRIES):
        delay = BASE_DELAY * (2 ** attempt)
        
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result["choices"][0]["message"]["content"].strip()
            
            if attempt > 0:
                print(f"SUCCESS (Attempt {attempt+1}): Translated after retry. Text: '{text[:30]}...'")
            return translated_text
        
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                print(f"FAILED (Max attempts exhausted): Text: '{text[:30]}...' Error: {e}")
                return f"[PARSING FAILED {e}]"
            else:
                print(f"FAILED (Attempt {attempt+1}). Retrying in {delay}s. Text: '{text[:30]}...' Error: {e}")
                time.sleep(delay)
        
        except (KeyError, IndexError) as e:
            print(f"FAILED (JSON PARSE): Text: '{text[:30]}...' Error: {e}")
            return f"[PARSING FAILED JSON Error: {e}]"
    
    return translated_text 


def translate_batch_via_api(batch, api_endpoint, headers, text_column):
    texts = batch[text_column]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        
        api_endpoints = [api_endpoint] * len(texts)
        all_headers = [headers] * len(texts)
        
        translated_texts = list(executor.map(
            translate_single_text, 
            texts, 
            api_endpoints, 
            all_headers
        ))
        
    batch["text_swahili"] = translated_texts
    return batch

# --- Function to run the job with CHUNKING and resuming ---
def run_translation_job(
        full_dataset, api_endpoint, headers, text_column, local_save_path, final_parquet_path, chunk_size):
    
    total_examples = len(full_dataset)
    os.makedirs(local_save_path, exist_ok=True)
    
    # Find existing checkpoints (chunks that were already translated)
    checkpoint_files = sorted(glob.glob(os.path.join(local_save_path, "chunk_*.arrow")))
    
    translated_chunks = []
    if checkpoint_files:
        # Load existing chunks to correctly determine the start index
        translated_chunks = [load_from_disk(f) for f in checkpoint_files]
        start_index = len(translated_chunks) * chunk_size
        print(f"RESUMING: Found {len(translated_chunks)} checkpoint chunks.")
        print(f"Total examples completed: {start_index}. Starting translation from index {start_index}.")
    else:
        start_index = 0
        print("STARTING NEW JOB: No prior checkpoints found.")
    
    # Loop through the dataset in chunks
    for i in range(start_index // chunk_size, (total_examples + chunk_size - 1) // chunk_size):
        
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, total_examples)
        
        # Determine the file path for the current chunk's checkpoint
        chunk_file_name = os.path.join(local_save_path, f"chunk_{i:04d}.arrow")
        
        if os.path.exists(chunk_file_name):
            # If the checkpoint file exists, skip processing
            print(f"Chunk {i} (Index {chunk_start} to {chunk_end-1}) already exists. Skipping.")
            continue

        print(f"\n--- Translating Chunk {i:04d} (Index {chunk_start} to {chunk_end-1}) ---")
        
        # Select the slice of the data for this chunk
        chunk_dataset = full_dataset.select(range(chunk_start, chunk_end))
        
        # Run the concurrent mapping on the chunk
        translated_chunk = chunk_dataset.map(
            lambda batch: translate_batch_via_api(batch, api_endpoint, headers, text_column),
            batched=True,
            batch_size=BATCH_SIZE,
            remove_columns=[], # Keep all columns for the checkpoint
        )
        
        # Save the translated chunk as a checkpoint
        translated_chunk.save_to_disk(chunk_file_name)
        print(f"Successfully saved checkpoint to: {chunk_file_name}")

    # --- Final Step: Merge all translated chunks and save final Parquet file ---
    print("\n--- All chunks processed. Merging final dataset ---")
    
    # Reload all saved chunks 
    final_checkpoint_files = sorted(glob.glob(os.path.join(local_save_path, "chunk_*.arrow")))
    
    if not final_checkpoint_files:
        print("Error: No translated chunks found to merge.")
        return

    # Load all individual translated chunks
    all_translated_chunks = [load_from_disk(f) for f in final_checkpoint_files]
    
    # Concatenate them into a single final dataset object
    final_translated_dataset = concatenate_datasets(all_translated_chunks)

    print(f"Final dataset merged: {len(final_translated_dataset)} examples.")
    
    # EXPORT TO PARQUET
    print(f"Exporting final merged dataset to Parquet format at: {final_parquet_path}")
    final_translated_dataset.to_parquet(final_parquet_path)
    print("Export complete.")


# --- Main Execution Block ---

# 1. Check if raw data exists locally. If not, download and save it.
if os.path.exists(RAW_DATA_PATH):
    print(f"Loading raw dataset from local disk: {RAW_DATA_PATH}")
    dataset = load_from_disk(RAW_DATA_PATH)
else:
    print(f"Downloading dataset { DATASET_NAME}, subset {SUBSET_NAME}, split {SPLIT}...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    print(f"Download complete. Saving raw dataset to local disk: {RAW_DATA_PATH}")
    dataset.save_to_disk(RAW_DATA_PATH)

print(f"Dataset loaded with {len(dataset)} examples. Starting translation job.")

# 2. Start the translation job using the locally loaded dataset
run_translation_job(
    dataset, 
    API_ENDPOINT, 
    HEADERS, 
    TEXT_COLUMN, 
    LOCAL_SAVE_PATH,
    FINAL_PARQUET_PATH,
    CHUNK_SIZE
)

print("Translation job finished.")
