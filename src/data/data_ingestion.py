import pandas as pd
from typing import Tuple
import yaml
from pathlib import Path
from src.utils.logger import setup_logger
from functools import wraps
import time

logger = setup_logger()

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

class DataIngestion:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.raw_data_path = self.config['data']['raw_data_path']
        
    @log_execution_time
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the ingested data."""
        try:
            # Check for required columns
            required_features = (
                self.config['features']['numeric_features'] +
                self.config['features']['categorical_features'] +
                [self.config['features']['target']]
            )
            
            for feature in required_features:
                assert feature in df.columns, f"Missing required feature: {feature}"
            
            # Check for null values
            assert not df.isnull().any().any(), "Dataset contains null values"
            
            logger.info("Data validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    @log_execution_time
    def load_data(self) -> pd.DataFrame:
        """Load data from the source."""
        try:
            logger.info(f"Loading data from {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)
            self.validate_data(df)
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    @log_execution_time
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        try:
            from sklearn.model_selection import train_test_split
            
            train_df, test_df = train_test_split(
                df,
                test_size=self.config['data']['train_test_split'],
                random_state=self.config['data']['random_state']
            )
            
            # Save split datasets
            processed_path = Path(self.config['data']['processed_data_path'])
            processed_path.mkdir(parents=True, exist_ok=True)
            
            train_df.to_csv(processed_path / "train.csv", index=False)
            test_df.to_csv(processed_path / "test.csv", index=False)
            
            logger.info("Data split completed and saved successfully")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise 