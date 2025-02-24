import numpy as np
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from src.utils.logger import setup_logger
from src.models.model_builder import ModelBuilder
import yaml
from pathlib import Path
import joblib

logger = setup_logger()

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_builder = ModelBuilder(config_path)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """Train the model and save it."""
        try:
            # Build and train model
            model = self.model_builder.build_model()
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                early_stopping_rounds=self.config['training']['early_stopping_rounds'],
                verbose=True
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config['training']['cross_validation_folds'],
                scoring='roc_auc'
            )
            
            logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            logger.info("Model training completed successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise 