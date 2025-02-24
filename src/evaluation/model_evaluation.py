import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import mlflow
import json
from src.utils.logger import setup_logger
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = setup_logger()

class ModelEvaluator:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model and return metrics."""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Log metrics to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_metrics({
                    "test_roc_auc": metrics["roc_auc"],
                    "test_accuracy": metrics["classification_report"]["accuracy"]
                })
                
                # Save confusion matrix plot
                self._plot_confusion_matrix(y_test, y_pred)
                
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plots_path = Path("reports/figures")
        plots_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_path / "confusion_matrix.png")
        plt.close() 