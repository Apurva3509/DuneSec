import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from src.utils.logger import setup_logger
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = setup_logger()

class TestPredictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load preprocessors
        self.load_preprocessors()
        
        # Define correlated feature pairs (same as in DataPreprocessor)
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
        
        # Get features to drop
        self.features_to_drop = set()
        for feature1, feature2 in self.correlated_feature_pairs:
            self.features_to_drop.add(feature2)
        
        # Update numeric features list
        self.numeric_features = [f for f in self.config['features']['numeric_features'] 
                               if f not in self.features_to_drop]
        
        # Load model
        self.model = joblib.load("models/model.joblib")
        
    def load_preprocessors(self):
        """Load saved preprocessors."""
        try:
            preprocessors_path = Path("models/preprocessors")
            self.scaler = joblib.load(preprocessors_path / "scaler.joblib")
            self.target_encoder = joblib.load(preprocessors_path / "target_encoder.joblib")
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise
    
    def predict_holdout_set(self):
        """
        Make predictions on the hold-out test set and evaluate performance.
        """
        try:
            # Load hold-out test set
            test_data = pd.read_csv("data/test/holdout_test_data.csv")
            
            # Drop the correlated features first
            test_data = test_data.drop(columns=self.features_to_drop, errors='ignore')
            
            # Ensure we only use the features that were used in training
            X_test = test_data[self.numeric_features]
            y_test = test_data[self.config['features']['target']]
            
            logger.info(f"Number of features in test set: {X_test.shape[1]}")
            logger.info(f"Features being used: {', '.join(self.numeric_features)}")
            
            # Preprocess features
            X_test_scaled = self.scaler.transform(X_test)
            y_test_encoded = self.target_encoder.transform(y_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test_encoded, y_pred, y_pred_proba)
            
            # Generate visualizations
            self.generate_test_visualizations(y_test_encoded, y_pred, y_pred_proba)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate performance metrics."""
        metrics = {
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        logger.info("\nHold-out Test Set Performance:")
        logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
        logger.info(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def generate_test_visualizations(self, y_true, y_pred, y_pred_proba):
        """Generate and save visualization for test set performance."""
        reports_path = Path("reports/test_results")
        reports_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Hold-out Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(reports_path / "test_confusion_matrix.png")
        plt.close()
        
        # Save detailed results
        with open(reports_path / "test_results.txt", 'w') as f:
            f.write("Hold-out Test Set Results\n")
            f.write("========================\n\n")
            f.write(f"Classification Report:\n{classification_report(y_true, y_pred)}\n")
            f.write(f"\nROC-AUC Score: {roc_auc_score(y_true, y_pred_proba):.4f}") 