import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
from src.utils.logger import setup_logger

logger = setup_logger()

def create_initial_split():
    """
    Perform initial split of data using configuration settings.
    """
    try:
        # Load configuration
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Load original dataset
        data_path = Path(config['data']['raw_data_path'])
        df = pd.read_csv(data_path)
        
        # Use split ratio from config
        test_size = config['data']['train_test_split']
        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=config['data']['random_state'],
            stratify=df[config['features']['target']]
        )
        
        # Create directories if they don't exist
        Path("data/train").mkdir(parents=True, exist_ok=True)
        Path("data/test").mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_data.to_csv("data/train/training_data.csv", index=False)
        test_data.to_csv("data/test/holdout_test_data.csv", index=False)
        
        logger.info(f"Initial split completed successfully:")
        logger.info(f"Training set size: {len(train_data)} samples")
        logger.info(f"Hold-out test set size: {len(test_data)} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in initial split: {str(e)}")
        raise 