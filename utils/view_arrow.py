import pyarrow.ipc as ipc
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


RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "/raid/.tnp/fineweb_edu_raw")
LOCAL_SAVE_PATH = os.getenv("LOCAL_SAVE_PATH", "/raid/.tnp/fineweb_edu_translated")
FINAL_PARQUET_PATH = os.getenv("FINAL_PARQUET_PATH", "/raid/.tnp/fineweb_edu_final.parquet")

def read_arrow():
    with open("/raid/.tnp/swa/fineweb_edu_translated/chunk_0010.arrow/data-00000-of-00001.arrow", "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()
        print(table)
        
    
def merge_arrow_files(local_save_path=LOCAL_SAVE_PATH, final_parquet_path=FINAL_PARQUET_PATH):
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


if __name__ == "__main__":
    merge_arrow_files()