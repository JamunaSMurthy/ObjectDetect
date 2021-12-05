"""
Logging configuration for ObjectDetect project.
Provides consistent logging across all modules with file and console output.
"""
import logging
import logging.handlers
from pathlib import Path
import os


def setup_logging(name, log_level=None, log_dir="logs"):
    """
    Configure logging for a module with both console and file handlers.
    
    Args:
        name (str): Module name (typically __name__)
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir (str): Directory to store log files
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / f"{name.replace('.', '_')}.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
