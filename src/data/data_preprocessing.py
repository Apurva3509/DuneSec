import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import yaml
from src.utils.logger import setup_logger
import joblib
from pathlib import Path

logger = setup_logger()

class DataPreprocessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.numeric_features = self.config['features']['numeric_features']
        self.target = self.config['features']['target']
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform the training data."""
        try:
            # Select only numeric features and target
            df = df[self.numeric_features + [self.target]].copy()
            
            # Handle any missing values
            df = self.handle_missing_values(df)
            
            X = df.copy()
            y = X.pop(self.target)
            
            # Encode target variable (BENIGN -> 0, DDoS -> 1)
            y = self.target_encoder.fit_transform(y)
            
            # Scale numeric features
            X = self.scaler.fit_transform(X)
            
            # Save preprocessors
            self._save_preprocessors()
            
            logger.info("Data preprocessing completed successfully")
            logger.info(f"Target classes mapping: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform the test data."""
        try:
            # Select only numeric features and target
            df = df[self.numeric_features + [self.target]].copy()
            
            # Handle any missing values
            df = self.handle_missing_values(df)
            
            X = df.copy()
            y = X.pop(self.target)
            
            # Encode target variable
            y = self.target_encoder.transform(y)
            
            # Scale numeric features
            X = self.scaler.transform(X)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with mean for numeric columns
        for col in df.columns:
            if col != self.target:
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _save_preprocessors(self):
        """Save the fitted preprocessors."""
        preprocessors_path = Path("models/preprocessors")
        preprocessors_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, preprocessors_path / "scaler.joblib")
        joblib.dump(self.target_encoder, preprocessors_path / "target_encoder.joblib") 