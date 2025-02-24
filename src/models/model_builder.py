import xgboost as xgb
from typing import Dict
import yaml
import mlflow
from src.utils.logger import setup_logger

logger = setup_logger()

class ModelBuilder:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_params = self.config['model']['params']
        
    def build_model(self) -> xgb.XGBClassifier:
        """Build and return an XGBoost classifier."""
        try:
            model = xgb.XGBClassifier(
                **self.model_params,
                random_state=self.config['data']['random_state']
            )
            
            logger.info("XGBoost model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise 