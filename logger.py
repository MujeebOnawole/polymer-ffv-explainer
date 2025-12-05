# logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

class LoggerSetup:
    """Centralized logger setup class"""
    _initialized = False
    _output_dir = None
    _task_name = None

    @classmethod
    def initialize(cls, output_dir: str, task_name: str):
        """Initialize logger settings"""
        cls._output_dir = output_dir
        cls._task_name = task_name
        cls._initialized = True

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Sets up and returns a logger with both console and rotating file handlers.
    
    Parameters
    ----------
    name : str, optional
        Name of the logger (typically __name__), by default __name__
        
    Returns
    -------
    logging.Logger
        Configured logger instance
        logger.debug() for detailed debugging information
        logger.info() for important status updates
        logger.warning() for concerning but non-fatal issues
        logger.error() for errors
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers to the logger
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Define a common formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler only if logger has been initialized with output directory
        if LoggerSetup._initialized:
            try:
                # Create log directory if it doesn't exist
                os.makedirs(LoggerSetup._output_dir, exist_ok=True)
                
                # Define log file path based on task name
                log_file = os.path.join(LoggerSetup._output_dir, f'{LoggerSetup._task_name}.log')
                
                # Create rotating file handler
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=10**6,
                    backupCount=5
                )
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Failed to set up file logging: {str(e)}")
    
    return logger
