"""
Custom logger for the Tourism Recommendation System.
Provides consistent, colored logging across all modules.
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Create and return a configured logger instance.
    
    Args:
        name: Name for the logger (usually __name__ of the calling module).
        log_file: Optional path to a log file. If None, logs to console only.
    
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)

    # ─── Console Handler ───
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # ─── File Handler (optional) ───
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(funcName)-20s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_project_logger(module_name: str) -> logging.Logger:
    """
    Convenience function that creates a logger with file logging
    enabled to the project's logs directory.
    
    Args:
        module_name: Name for the logger (usually __name__).
    
    Returns:
        Configured logger that writes to both console and log file.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"tourism_rec_{today}.log")
    
    return get_logger(module_name, log_file)
