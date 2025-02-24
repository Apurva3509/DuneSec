from sklearn.ensemble import RandomForestClassifier
from typing import Dict
import yaml
import mlflow
from src.utils.logger import setup_logger

logger = setup_logger()

class ModelBuilder:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': self.config['data']['random_state'],
            'n_jobs': -1
        }
        
    def build_model(self) -> RandomForestClassifier:
        """Build and return a Random Forest classifier."""
        try:
            model = RandomForestClassifier(**self.model_params)
            
            logger.info("Random Forest model built successfully")
            logger.info(f"Model parameters: {self.model_params}")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise 