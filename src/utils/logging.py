"""
Logging utilities for OSPA experiments.
"""

import os
import logging
import sys
from datetime import datetime


def setup_logger(log_dir, name="ospa"):
    """
    Set up a logger that writes to both console and a file.
    
    Args:
        log_dir: Directory to save log file
        name: Logger name
        
    Returns:
        logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set formatters
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to {log_file}")
    
    return logger


def get_logger(name="ospa"):
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        logger: Logger instance
    """
    return logging.getLogger(name)