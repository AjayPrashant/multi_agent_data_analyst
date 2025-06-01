# agents/modeling_agent.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class ModelingAgent:
    def __init__(self, model_output_folder='outputs/models'):
        self.model_output_folder = model_output_folder
        os.makedirs(self.model_output_folder, exist_ok=True)

    def detect_target_column(self, data: pd.DataFrame) -> str:
        # Priority: if 'target' exists, use it
        if 'target' in data.columns:
            logger.info("Detected target column: 'target'")
            return 'target'

        # Else: check last column
        potential_target = data.columns[-1]
        if data[potential_target].dtype == 'object' or data[potential_target].nunique() < 10:
            logger.info(f"Auto-detected target column: '{potential_target}'")
            return potential_target

        raise ValueError("Unable to auto-detect target column. Please specify it manually.")

    def run(self, data: pd.DataFrame):
        logger.info("🤖 Modeling Agent: Training model...")

        target_column = self.detect_target_column(data)

        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Encode target if needed
        if y.dtype == 'object':
            logger.info("Encoding target column...")
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(self.model_output_folder, "model.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")

        # Cross-validation
        logger.info("Performing 5-fold cross-validation...")
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation scores: {scores}")
        logger.info(f"Mean CV accuracy: {scores.mean():.4f}")

        # Return model and test set for EvaluationAgent
        return model, X_test, y_test
