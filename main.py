import argparse
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.evaluation.model_evaluation import ModelEvaluator
import mlflow
from src.utils.logger import setup_logger
import yaml

logger = setup_logger()

def main(mode: str):
    """Main execution function."""
    try:
        # Load configuration
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        if mode in ['train', 'all']:
            # Data ingestion
            logger.info("Starting data ingestion...")
            data_ingestion = DataIngestion()
            df = data_ingestion.load_data()
            train_df, test_df = data_ingestion.split_data(df)
            
            # Data preprocessing
            logger.info("Starting data preprocessing...")
            preprocessor = DataPreprocessor()
            X_train, y_train = preprocessor.fit_transform(train_df)
            X_test, y_test = preprocessor.transform(test_df)
            
            # Model training
            logger.info("Starting model training...")
            trainer = ModelTrainer()
            trainer.train(X_train, y_train)
            
            # Model evaluation
            logger.info("Starting model evaluation...")
            evaluator = ModelEvaluator()
            model = mlflow.xgboost.load_model("models/model.xgb")
            metrics = evaluator.evaluate(model, X_test, y_test)
            
            logger.info(f"Final ROC-AUC Score: {metrics['roc_auc']:.4f}")
        
        if mode == 'predict':
            # Load model and make predictions
            logger.info("Loading model for predictions...")
            model = mlflow.xgboost.load_model("models/model.xgb")
            # Add prediction logic here
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDoS Detection System")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['train', 'predict', 'all'],
        default='all',
        help="Execution mode: train, predict, or all"
    )
    args = parser.parse_args()
    main(args.mode) 