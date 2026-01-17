"""
Centralized logging configuration for the capacity optimization system.

This module provides consistent logging setup across all components with:
- Console output for real-time monitoring
- File logging for error tracking and debugging
- Configurable log levels
- Structured log formatting
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os


def setup_logger(
    name: str, 
    level: str = "INFO", 
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to write logs to file
        log_to_console: Whether to write logs to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler for real-time monitoring
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    
    # File handler for persistent logging
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # General log file
        file_handler = logging.FileHandler(log_dir / "capacity_optimizer.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        
        # Error-only log file
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with default configuration.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    # Check environment variables for configuration
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_to_file = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    log_to_console = os.getenv('LOG_TO_CONSOLE', 'true').lower() == 'true'
    
    return setup_logger(
        name=name,
        level=log_level,
        log_to_file=log_to_file,
        log_to_console=log_to_console
    )


def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log function entry with parameters.
    
    Args:
        logger: Logger instance
        func_name: Function name
        **kwargs: Function parameters to log
    """
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Entering {func_name}({params})")


def log_function_exit(logger: logging.Logger, func_name: str, result: Optional[str] = None):
    """
    Log function exit with optional result summary.
    
    Args:
        logger: Logger instance
        func_name: Function name
        result: Optional result summary
    """
    if result:
        logger.debug(f"Exiting {func_name}: {result}")
    else:
        logger.debug(f"Exiting {func_name}")


def log_data_summary(logger: logging.Logger, data_name: str, count: int, summary: str = ""):
    """
    Log data loading/processing summary.
    
    Args:
        logger: Logger instance
        data_name: Name of the data being processed
        count: Number of records/items
        summary: Optional summary description
    """
    message = f"Loaded {data_name}: {count} records"
    if summary:
        message += f" - {summary}"
    logger.info(message)


def log_optimization_progress(logger: logging.Logger, stage: str, progress: str):
    """
    Log optimization progress updates.
    
    Args:
        logger: Logger instance
        stage: Current optimization stage
        progress: Progress description
    """
    logger.info(f"[{stage}] {progress}")


def log_validation_result(logger: logging.Logger, validation_type: str, passed: bool, details: str = ""):
    """
    Log validation results consistently.
    
    Args:
        logger: Logger instance
        validation_type: Type of validation performed
        passed: Whether validation passed
        details: Additional validation details
    """
    status = "PASSED" if passed else "FAILED"
    message = f"Validation {validation_type}: {status}"
    if details:
        message += f" - {details}"
    
    if passed:
        logger.info(message)
    else:
        logger.warning(message)