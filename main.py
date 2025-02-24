import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import yaml
from src.data.data_preprocessing import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.evaluation.model_evaluation import ModelEvaluator
from src.utils.logger import setup_logger
import datetime
from sklearn.model_selection import train_test_split

logger = setup_logger()

def load_config():
    """Load configuration from yaml file."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_initial_split():
    """Create initial train-test split."""
    try:
        # Load the full dataset
        df = pd.read_csv("data/raw/network_traffic.csv")
        
        # Perform stratified split
        train_df = df.sample(frac=0.9, random_state=42)
        test_df = df.drop(train_df.index)
        
        # Create directories if they don't exist
        Path("data/train").mkdir(parents=True, exist_ok=True)
        Path("data/test").mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_df.to_csv("data/train/training_data.csv", index=False)
        test_df.to_csv("data/test/testing_data.csv", index=False)
        
        logger.info(f"Split completed. Training set size: {len(train_df)}, Test set size: {len(test_df)}")
        
    except Exception as e:
        logger.error(f"Error in creating data split: {str(e)}")
        raise

def main(mode: str):
    """Main execution function."""
    try:
        if mode == 'split':
            logger.info("Loading and splitting data...")
            
            # Load config and data
            config = load_config()
            df = pd.read_csv("data/raw/network_traffic.csv")
            
            preprocessor = DataPreprocessor()
            X, y = preprocessor.fit_transform(df)
            
            # Create train-test split using config value
            test_size = config['data']['train_test_split']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size,
                stratify=y, 
                random_state=config['data']['random_state']
            )
            
            # Save splits with proper index=False to avoid extra columns
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            
            # Save training data
            train_df = pd.concat([X_train, pd.Series(y_train, name='Label')], axis=1)
            train_df.to_csv("data/processed/train.csv", index=False, float_format='%.6f')
            
            # Save test data
            test_df = pd.concat([X_test, pd.Series(y_test, name='Label')], axis=1)
            test_df.to_csv("data/processed/test.csv", index=False, float_format='%.6f')
            
            logger.info(f"Split completed. Training set size: {len(train_df)}, Test set size: {len(test_df)}")
            logger.info(f"Using test_size={test_size} from config")
            
        elif mode == 'train':
            # Load and validate training data
            train_df = pd.read_csv("data/processed/train.csv")
            
            # Clean any potential NaN/infinite values that might have been introduced
            train_df = train_df.replace([np.inf, -np.inf], np.nan)
            train_df = train_df.dropna()
            
            # Split features and target
            X_train = train_df.drop('Label', axis=1)
            y_train = train_df['Label']
            
            # Verify no NaN/infinite values
            assert not X_train.isnull().any().any(), "NaN values found in features"
            assert not np.isinf(X_train.values).any(), "Infinite values found in features"
            assert not y_train.isnull().any(), "NaN values found in target"
            
            logger.info(f"Training data loaded successfully. Shape: {X_train.shape}")
            logger.info(f"Class distribution:\n{pd.Series(y_train).value_counts(normalize=True)}")
            
            # Train model
            trainer = ModelTrainer()
            model = trainer.train(X_train, y_train)
            
            # Save model
            Path("models").mkdir(exist_ok=True)
            joblib.dump(model, "models/random_forest_model.joblib")
            
            # Evaluate on training set
            evaluator = ModelEvaluator()
            evaluator.evaluate(model, X_train, y_train, "train")
            
        elif mode == 'test':
            # Load and validate test data
            test_df = pd.read_csv("data/processed/test.csv")
            
            # Clean any potential NaN/infinite values
            test_df = test_df.replace([np.inf, -np.inf], np.nan)
            test_df = test_df.dropna()
            
            # Split features and target
            X_test = test_df.drop('Label', axis=1)
            y_test = test_df['Label']
            
            # Verify no NaN/infinite values
            assert not X_test.isnull().any().any(), "NaN values found in features"
            assert not np.isinf(X_test.values).any(), "Infinite values found in features"
            assert not y_test.isnull().any(), "NaN values found in target"
            
            # Load model
            model = joblib.load("models/random_forest_model.joblib")
            
            # Evaluate on test set
            evaluator = ModelEvaluator()
            evaluator.evaluate(model, X_test, y_test, "test")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['split', 'train', 'test'],
                      help='Mode of operation: split, train, or test')
    args = parser.parse_args()
    main(args.mode) 