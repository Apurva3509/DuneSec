import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import yaml
from src.utils.logger import setup_logger
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = setup_logger()

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
        self.categorical_features = ["Source Port", "Destination Port", "Protocol"]
        self.correlated_feature_pairs = [
            ("Active Max", "Active Min"),
            ("Average Packet Size", "Packet Length Mean"),
            ("Avg Bwd Segment Size", "Avg Fwd Segment Size"),
            ("Bwd Header Length", "Fwd Header Length"),
            ("Bwd IAT Max", "Bwd IAT Min"),
            ("Bwd Packet Length Mean", "Bwd Packet Length Std"),
            ("Flow IAT Max", "Flow IAT Std"),
            ("Fwd IAT Max", "Fwd IAT Total"),
            ("Fwd IAT Mean", "Fwd IAT Std"),
            ("Fwd Packet Length Mean", "Fwd Packet Length Std"),
            ("Idle Max", "Idle Min"),
            ("Subflow Bwd Packets", "Subflow Fwd Packets")
        ]
        self.param_grid = {
            'n_estimators': [100],               # Reduced from [100, 200]
            'max_depth': [None, 20],             # Reduced from [None, 20, 30]
            'min_samples_split': [2],            # Reduced from [2, 5]
            'min_samples_leaf': [1],             # Reduced from [1, 2]
            'max_features': ['sqrt']             # Reduced from ['sqrt', 'log2']
        }

    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN and then handle them."""
        # Replace inf and -inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        try:
            # Create a copy
            df = df.copy()
            
            # First handle infinite values
            df = self._handle_infinite_values(df)
            
            # Handle numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # Get median of finite values
                median_val = df[col][np.isfinite(df[col])].median()
                # Replace NaN and infinite values with median
                df[col] = df[col].fillna(median_val)
            
            # Handle categorical columns
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'UNKNOWN'
                df[col] = df[col].fillna(mode_val)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in handling missing values: {str(e)}")
            raise

    def fit_transform(self, df: pd.DataFrame):
        """Fit and transform the data preprocessing pipeline."""
        try:
            logger.info("Starting data preprocessing...")
            
            # Create a copy to avoid modifying original data
            df = df.copy()
            
            # Encode labels first
            df["Label"] = self.label_encoder.fit_transform(df["Label"])
            
            # Drop non-relevant columns
            df.drop(columns=self.drop_cols, errors="ignore", inplace=True)
            
            # Handle categorical features using frequency encoding - None as of now, but can be used if any in future
            for col in self.categorical_features:
                if col in df.columns:
                    freq_map = df[col].value_counts().to_dict()
                    df[col] = df[col].map(freq_map)
            
            # Drop highly correlated features
            features_to_drop = set()
            for feature1, feature2 in self.correlated_feature_pairs:
                features_to_drop.add(feature2)
            df.drop(columns=features_to_drop, errors="ignore", inplace=True)
            
            # Split features and target
            X = df.drop("Label", axis=1)
            y = df["Label"]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            logger.info(f"Data preprocessing completed. Features shape: {X_scaled.shape}")
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted preprocessing pipeline."""
        try:
            df = df.copy()
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle inf/nan values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Drop non-relevant columns
            df.drop(columns=self.drop_cols, errors="ignore", inplace=True)
            
            # Handle categorical features
            for col in self.categorical_features:
                if col in df.columns:
                    freq_map = df[col].value_counts().to_dict()
                    df[col] = df[col].map(freq_map)
            
            # Drop highly correlated features
            features_to_drop = set()
            for feature1, feature2 in self.correlated_feature_pairs:
                features_to_drop.add(feature2)
            df.drop(columns=features_to_drop, errors="ignore", inplace=True)
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            X_scaled = pd.DataFrame(X_scaled, columns=df.columns)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise

    def _save_preprocessors(self):
        """Save the fitted preprocessors."""
        preprocessors_path = Path("models/preprocessors")
        preprocessors_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, preprocessors_path / "scaler.joblib")
        joblib.dump(self.label_encoder, preprocessors_path / "label_encoder.joblib")

    def create_initial_split(self, df: pd.DataFrame):
        """Create train-test split with proper data preprocessing."""
        try:
            logger.info("Starting data preprocessing and split...")
            
            # Create a copy
            df = df.copy()
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle inf/nan values
            initial_shape = df.shape
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            logger.info(f"Removed {initial_shape[0] - df.shape[0]} rows with missing values")
            
            # Remove duplicates
            initial_shape = df.shape
            df.drop_duplicates(inplace=True)
            logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
            
            # Log class distribution
            logger.info("\nLabel Distribution:")
            logger.info(df["Label"].value_counts())
            logger.info("\nLabel Distribution (%):")
            logger.info(df["Label"].value_counts(normalize=True) * 100)
            
            # Encode labels
            df["Label"] = self.label_encoder.fit_transform(df["Label"])
            
            # Drop non-relevant columns
            df.drop(columns=self.drop_cols, errors="ignore", inplace=True)
            
            # Handle categorical features
            for col in self.categorical_features:
                if col in df.columns:
                    freq_map = df[col].value_counts().to_dict()
                    df[col] = df[col].map(freq_map)
            
            # Drop highly correlated features
            features_to_drop = set()
            for feature1, feature2 in self.correlated_feature_pairs:
                features_to_drop.add(feature2)
            df.drop(columns=features_to_drop, errors="ignore", inplace=True)
            
            # Split features and target
            X = df.drop("Label", axis=1)
            y = df["Label"]
            
            # Load config
            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Use split ratio from config
            test_size = config['data']['train_test_split']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrames
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            logger.info(f"Final training set shape: {X_train_scaled.shape}")
            logger.info(f"Final test set shape: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in creating data split: {str(e)}")
            raise 