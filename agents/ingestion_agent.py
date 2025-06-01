# agents/ingestion_agent.py

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


class IngestionAgent:
    def __init__(self, data_folder='docs'):
        self.data_folder = data_folder

    def run(self):
        print("📥 Ingestion Agent: Loading data...")
        
        # For simplicity, load first CSV in folder
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in docs/ folder.")
        
        file_path = os.path.join(self.data_folder, csv_files[0])
        print(f"Loading: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Data Shape: {df.shape}")
        
        return df
