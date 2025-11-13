import os
from datasets import load_dataset
import pandas as pd
import logging 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSetProcessor:
    def __init__(self, parquet_dir, repo_id: str, lang: str):
        self.parquet_dir = parquet_dir
        self.repo_id = repo_id
        
        self.new_column_order = [
            'text', f'text_swahili', 'id', 'dump', 'url', 'file_path', 
            'language', 'language_score', 'token_count', 'score', 'int_score'
        ]
        
    def _check_parquet_files(self):
        if not os.path.exists(self.parquet_dir):
            raise FileNotFoundError(f"Directory {self.parquet_dir} not found")
            
        parquet_files = [
            os.path.join(self.parquet_dir, f) 
            for f in os.listdir(self.parquet_dir) 
            if f.endswith('.parquet')
        ]
        return parquet_files
    
    def load_prep_datasets(self):
        try:
            parquet_files = self._check_parquet_files()
            self.dataset = load_dataset('parquet', data_files=parquet_files)
            current_columns = self.dataset['train'].column_names
            self.dataset['train'] = self.dataset['train'].select_columns(self.new_column_order)
        except Exception as e:
            raise

    def push_data(self):
        if self.dataset is None:
            raise ValueError("The dataset is not loaded")
        try:
            self.dataset.push_to_hub(self.repo_id)
            print("Successfuly pushed to hub")
        except Exception as e:
            print(f"Error pushing to hub")


if __name__ == '__main__':
    LANG = "luo"
    REPO_ID = f"rao254/test-swa"
    PARQUET_DIR = f"/raid/.tnp/swa/"
    data = DataSetProcessor(parquet_dir=PARQUET_DIR, repo_id=REPO_ID, lang=LANG)   
    data.load_prep_datasets()
    data.push_data()
