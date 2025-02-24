import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import yaml

def setup_logger(config_path: str = "config/config.yaml"):
    """Set up logger with configuration from yaml file."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    log_path = config['logging']['log_path']
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('DDosDetection')
    logger.setLevel(config['logging']['level'])
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # File Handler
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger 