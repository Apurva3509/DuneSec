import argparse
import pandas as pd
import mlflow
import mlflow.xgboost
from src.data.initial_split import create_initial_split
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.evaluation.model_evaluation import ModelEvaluator
from src.evaluation.test_predictor import TestPredictor
from src.utils.logger import setup_logger
import xgboost as xgb
import joblib
from pathlib import Path

logger = setup_logger()

def main(mode: str):
    """Main execution function."""
    try:
        if mode == 'split':
            # Perform initial train-test split
            logger.info("Performing initial 90-10 data split...")
            create_initial_split()
            
        elif mode == 'train':
            # Train model using training data
            logger.info("Loading training data...")
            data_ingestion = DataIngestion()
            train_df = pd.read_csv("data/train/training_data.csv")
            
            # Data preprocessing
            logger.info("Starting data preprocessing...")
            preprocessor = DataPreprocessor()
            X_train, y_train = preprocessor.fit_transform(train_df)
            
            # Model training
            logger.info("Starting model training...")
            trainer = ModelTrainer()
            model = trainer.train(X_train, y_train)
            
            # Save model using joblib instead of mlflow
            model_path = Path("models")
            model_path.mkdir(exist_ok=True)
            joblib.dump(model, model_path / "model.joblib")
            
            # Model evaluation
            logger.info("Starting model evaluation...")
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(model, X_train, y_train)
            
            logger.info(f"Final ROC-AUC Score: {metrics['roc_auc']:.4f}")
            
        elif mode == 'test':
            # Evaluate on hold-out test set
            logger.info("Evaluating model on hold-out test set...")
            predictor = TestPredictor()
            metrics = predictor.predict_holdout_set()
            
        elif mode == 'predict':
            # Make predictions on new data
            logger.info("Making predictions on new data...")
            # [Add your prediction code here]
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDoS Detection System")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['split', 'train', 'test', 'predict'],
        required=True,
        help="Execution mode: split (initial split), train (train model), test (evaluate on hold-out), predict (new data)"
    )
    args = parser.parse_args()
    main(args.mode) 