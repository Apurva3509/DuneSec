import numpy as np
from sklearn.metrics import (roc_curve, auc, classification_report, confusion_matrix,
                           accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
import mlflow
import json
from src.utils.logger import setup_logger
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import pandas as pd

logger = setup_logger()

class ModelEvaluator:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.reports_path = Path("reports")
        self.figures_path = self.reports_path / "figures"
        self.results_path = self.reports_path / "results"
        
        # Create directories if they don't exist
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, model, X, y, dataset_type="train"):
        """Evaluate model performance and generate plots."""
        try:
            metrics = {}
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Measure prediction time
            start_time = time.time()
            y_pred = model.predict(X)
            end_time = time.time()
            prediction_time = (end_time - start_time) / len(X) * 1000  # ms per sample
            
            # Get prediction probabilities
            y_prob = model.predict_proba(X)
            
            # Calculate metrics
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['precision'] = precision_score(y, y_pred)
            metrics['recall'] = recall_score(y, y_pred)
            metrics['f1'] = f1_score(y, y_pred)
            metrics['roc_auc'] = roc_auc_score(y, y_prob[:, 1])
            metrics['avg_prediction_time_ms'] = prediction_time
            
            # Generate plots
            self._plot_prediction_probabilities(y_prob, dataset_type, timestamp)
            self._plot_threshold_performance(y, y_prob[:, 1], dataset_type, timestamp)
            self._plot_roc_curve(y, y_prob[:, 1], dataset_type, timestamp)
            self._plot_confusion_matrix(y, y_pred, dataset_type, timestamp)
            
            if hasattr(model, 'feature_importances_'):
                self._plot_feature_importance(model, X.columns, dataset_type, timestamp)
            
            # Save metrics to JSON
            results_file = self.results_path / f"{dataset_type}_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'metrics': {
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1'],
                        'roc_auc': metrics['roc_auc'],
                        'avg_prediction_time_ms': metrics['avg_prediction_time_ms'],
                        'timestamp': timestamp,
                        'dataset_type': dataset_type,
                        'n_samples': len(y),
                        'n_features': X.shape[1]
                    },
                    'classification_report': classification_report(y, y_pred)
                }, f, indent=4)
            
            logger.info(f"\nEvaluation results for {dataset_type} set:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Average prediction time: {metrics['avg_prediction_time_ms']:.3f} ms per sample")
            logger.info(f"\nResults saved to {results_file}")
            logger.info(f"Plots saved in {self.figures_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
    
    def _plot_prediction_probabilities(self, y_prob, dataset_type, timestamp):
        """Plot distribution of prediction probabilities."""
        plt.figure(figsize=(10, 6))
        plt.hist(y_prob[:, 1], bins=50, color='skyblue', edgecolor='black')  # Plot probabilities for DDoS class
        plt.title(f'Distribution of DDoS Prediction Probabilities - {dataset_type.capitalize()} Set')
        plt.xlabel('Probability of DDoS Class')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(self.figures_path / f"prediction_dist_{dataset_type}_{timestamp}.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory
        
        logger.info(f"Saved prediction distribution plot to: prediction_dist_{dataset_type}_{timestamp}.png")

    def _plot_threshold_performance(self, y_true, y_prob, dataset_type, timestamp):
        """Plot performance metrics vs probability threshold."""
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_prob >= threshold).astype(int)
            threshold_metrics.append({
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred_threshold),
                'precision': precision_score(y_true, y_pred_threshold),
                'recall': recall_score(y_true, y_pred_threshold),
                'f1': f1_score(y_true, y_pred_threshold)
            })
        
        metrics_df = pd.DataFrame(threshold_metrics)
        plt.figure(figsize=(10, 6))
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plt.plot(metrics_df['threshold'], metrics_df[metric], label=metric)
        plt.xlabel('Probability Threshold')
        plt.ylabel('Score')
        plt.title(f'Performance Metrics vs Threshold - {dataset_type.capitalize()} Set')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figures_path / f"threshold_performance_{dataset_type}_{timestamp}.png")
        plt.close()

    def _plot_roc_curve(self, y_true, y_prob, dataset_type, timestamp):
        """Plot ROC curve."""
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_type.capitalize()} Set')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(self.figures_path / f"roc_curve_{dataset_type}_{timestamp}.png")
        plt.close()
        
    def _plot_confusion_matrix(self, y_true, y_pred, dataset_type, timestamp):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['BENIGN', 'DDoS'],
                   yticklabels=['BENIGN', 'DDoS'])
        plt.title(f'Confusion Matrix - {dataset_type.capitalize()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.figures_path / f"confusion_matrix_{dataset_type}_{timestamp}.png")
        plt.close()
        
    def _plot_feature_importance(self, model, X, dataset_type, timestamp):
        """Plot feature importance."""
        plt.figure(figsize=(12, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f'Feature Importances - {dataset_type.capitalize()} Set')
        plt.bar(range(X.shape[0]), importances[indices])
        plt.xticks(range(X.shape[0]), X[indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(self.figures_path / f"feature_importance_{dataset_type}_{timestamp}.png")
        plt.close() 