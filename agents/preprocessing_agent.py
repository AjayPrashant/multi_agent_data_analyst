# agents/preprocessing_agent.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        print("🧹 Preprocessing Agent: Cleaning data...")

        # Example: Impute numeric columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        imputer = SimpleImputer(strategy='mean')
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        
        # Example: Standardize numeric columns
        scaler = StandardScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        print("Preprocessing complete.")
        return data
