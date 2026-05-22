"""
Logging configuration for RAG Agent
"""
import logging
import sys
from pathlib import Path
from src.config import settings


def setup_logger(name: str = "rag_agent") -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)

    # Create logs directory if it doesn't exist
    settings.create_directories()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Create default logger
logger = setup_logger()
