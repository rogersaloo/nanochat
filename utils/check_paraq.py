import os
from datasets import load_from_disk

# --- Configuration (Update as needed) ---

# The directory where your chunks are saved (from concurrent_translator.py)
CHECKPOINT_DIR = "/raid/.tnp"

# The specific chunk file you want to inspect (e.g., the very first one)
# NOTE: This path should point to the .arrow directory created inside CHECKPOINT_DIR
CHUNK_TO_VIEW = os.path.join(CHECKPOINT_DIR, "chunk_0000.arrow")

# --- Viewer Logic ---

if not os.path.exists(CHUNK_TO_VIEW):
    print(f"Error: Could not find checkpoint file or directory at '{CHUNK_TO_VIEW}'.")
    print("Ensure you have run concurrent_translator.py at least once and that the path is correct.")
else:
    try:
        # Load the checkpoint chunk back into a Dataset object
        print(f"Loading checkpoint from: {CHUNK_TO_VIEW}")
        checkpoint_dataset = load_from_disk(CHUNK_TO_VIEW)
        
        print(f"\n--- Dataset Info ---")
        print(f"Number of examples: {len(checkpoint_dataset)}")
        print(f"Features (Columns): {checkpoint_dataset.features}")
        
        # Convert the first 5 examples to a Pandas DataFrame for easy viewing
        # This is a common practice for quick inspection in the terminal
        print("\n--- First 5 Examples (as Pandas DataFrame) ---")
        df = checkpoint_dataset.select(range(min(5, len(checkpoint_dataset)))).to_pandas()
        print(df.to_markdown(index=False))

    except Exception as e:
        print(f"\nAn error occurred while loading or viewing the file: {e}")
