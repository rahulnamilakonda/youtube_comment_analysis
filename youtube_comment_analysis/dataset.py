from pathlib import Path
import kagglehub
from loguru import logger
import pandas as pd
import os
import tempfile
import yaml
from youtube_comment_analysis.config import RAW_DATA_DIR

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def get_raw_data(output_dir: Path):
    params = load_params()
    datasets = params['data_collection']['datasets']
    
    for name, url in datasets.items():
        logger.info(f"Downloading {name} dataset from Kaggle: {url}")    
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:   
                dataset_path = kagglehub.dataset_download(url, output_dir=tmp_dir)
                files = os.listdir(dataset_path)
                csv_files = [f for f in files if f.endswith(".csv")]

                if not csv_files:
                    logger.warning(f"No CSV file found in {name} dataset. Checking subdirectories...")
                    # Some datasets have subdirs
                    for root, dirs, files in os.walk(dataset_path):
                        csv_files = [os.path.join(root, f) for f in files if f.endswith(".csv")]
                        if csv_files: break
                else:
                    csv_files = [os.path.join(dataset_path, f) for f in csv_files]

                if not csv_files:
                    logger.error(f"No CSV file found in {name} dataset") 
                    continue

                # Save all CSVs from the dataset (some have multiple like train/test)
                output_subdir = output_dir / name
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                for csv_f in csv_files:
                    data = pd.read_csv(csv_f, on_bad_lines='skip', engine='python')
                    save_name = os.path.basename(csv_f)
                    data.to_csv(output_subdir / save_name, index=False)
                
                logger.success(f"Dataset {name} saved to {output_subdir}") 
            
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")

if __name__ == "__main__":
    get_raw_data(RAW_DATA_DIR)
