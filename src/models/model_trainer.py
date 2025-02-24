import numpy as np
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.xgboost
from src.utils.logger import setup_logger
from src.models.model_builder import ModelBuilder
import yaml
from pathlib import Path
import os

logger = setup_logger()

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_builder = ModelBuilder(config_path)
        
        # Set up MLflow with local directory
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        
        # Create experiment or get existing one
        try:
            mlflow.create_experiment(self.config['mlflow']['experiment_name'])
        except Exception:
            pass
        
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model and log metrics with MLflow."""
        try:
            with mlflow.start_run():
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
                
                # Log metrics
                mlflow.log_params(self.config['model']['params'])
                mlflow.log_metric("cv_score_mean", cv_scores.mean())
                mlflow.log_metric("cv_score_std", cv_scores.std())
                
                # Save model
                model_path = Path("models")
                model_path.mkdir(exist_ok=True)
                mlflow.xgboost.save_model(model, model_path / "model.xgb")
                
                logger.info("Model training completed successfully")
                
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise 