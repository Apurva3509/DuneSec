import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
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
        self.reports_path = Path("reports/figures")
        self.reports_path.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, model, X_train, y_train, X_test=None, y_test=None):
        """Evaluate model performance and generate plots."""
        try:
            metrics = {}
            
            # Training metrics
            y_train_pred = model.predict(X_train)
            y_train_prob = model.predict_proba(X_train)[:, 1]
            
            # Calculate ROC-AUC score
            metrics['roc_auc'] = roc_auc_score(y_train, y_train_prob)
            metrics['train_report'] = classification_report(y_train, y_train_pred)
            
            # Generate and save training plots
            self._plot_roc_curve(y_train, y_train_prob, 'train')
            self._plot_confusion_matrix(y_train, y_train_pred, 'train')
            self._plot_feature_importance(model, X_train, 'feature_importance')
            
            # If test data is provided
            if X_test is not None and y_test is not None:
                y_test_pred = model.predict(X_test)
                y_test_prob = model.predict_proba(X_test)[:, 1]
                
                metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_prob)
                metrics['test_report'] = classification_report(y_test, y_test_pred)
                
                # Generate and save test plots
                self._plot_roc_curve(y_test, y_test_prob, 'test')
                self._plot_confusion_matrix(y_test, y_test_pred, 'test')
                
                # Compare train vs test ROC curves
                self._plot_train_test_comparison(
                    y_train, y_train_prob,
                    y_test, y_test_prob
                )
            
            logger.info("Evaluation plots saved in reports/figures/")
            logger.info(f"Training ROC-AUC Score: {metrics['roc_auc']:.4f}")
            if 'test_roc_auc' in metrics:
                logger.info(f"Testing ROC-AUC Score: {metrics['test_roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
    
    def _plot_roc_curve(self, y_true, y_prob, dataset_type):
        """Plot ROC curve."""
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_type.capitalize()} Set')
        plt.legend(loc="lower right")
        plt.savefig(self.reports_path / f"roc_curve_{dataset_type}.png")
        plt.close()
        
    def _plot_confusion_matrix(self, y_true, y_pred, dataset_type):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_type.capitalize()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.reports_path / f"confusion_matrix_{dataset_type}.png")
        plt.close()
        
    def _plot_feature_importance(self, model, X_train, filename):
        """Plot feature importance."""
        plt.figure(figsize=(12, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title("Feature Importances")
        plt.bar(range(X_train.shape[1]), importances[indices])
        plt.xticks(range(X_train.shape[1]), indices, rotation=45)
        plt.tight_layout()
        plt.savefig(self.reports_path / f"{filename}.png")
        plt.close()
        
    def _plot_train_test_comparison(self, y_train, y_train_prob, y_test, y_test_prob):
        """Plot training vs test ROC curves."""
        plt.figure(figsize=(10, 8))
        
        # Training ROC
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
        roc_auc_train = auc(fpr_train, tpr_train)
        plt.plot(fpr_train, tpr_train, 
                label=f'Training (AUC = {roc_auc_train:.2f})')
        
        # Testing ROC
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
        roc_auc_test = auc(fpr_test, tpr_test)
        plt.plot(fpr_test, tpr_test, 
                label=f'Testing (AUC = {roc_auc_test:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Training vs Testing')
        plt.legend(loc="lower right")
        plt.savefig(self.reports_path / "roc_curve_comparison.png")
        plt.close() 