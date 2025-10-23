"""
Logging configuration module for the Multimodal Document Search project.
Provides centralized logging setup with file and console handlers.

Environment variables:
- LOG_LEVEL: Override default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- LOG_FILE_LEVEL: Override file handler log level
- LOG_CONSOLE_LEVEL: Override console handler log level
- LOG_FILE_SIZE: Max log file size in bytes (default: 10MB)
- LOG_BACKUP_COUNT: Number of backup log files (default: 5)
- PERFORMANCE_LOGGING: Enable/disable performance logging (TRUE/FALSE)
"""

import os
import logging
import time
import functools
from logging.handlers import RotatingFileHandler
from datetime import datetime

# --- Logger Configuration ---
# These can be overridden with environment variables
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_FILE_LOG_LEVEL = logging.DEBUG
DEFAULT_CONSOLE_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
DEFAULT_LOG_BACKUP_COUNT = 5
DEFAULT_PERFORMANCE_LOGGING = True  # Enable performance logging by default

# Get log level from environment or use default
def _get_log_level(env_var, default_level):
    level_name = os.environ.get(env_var)
    if level_name:
        try:
            return getattr(logging, level_name.upper())
        except AttributeError:
            print(f"Invalid log level {level_name}. Using default.")
    return default_level

LOG_LEVEL = _get_log_level('LOG_LEVEL', DEFAULT_LOG_LEVEL)
FILE_LOG_LEVEL = _get_log_level('LOG_FILE_LEVEL', DEFAULT_FILE_LOG_LEVEL)
CONSOLE_LOG_LEVEL = _get_log_level('LOG_CONSOLE_LEVEL', DEFAULT_CONSOLE_LOG_LEVEL)
LOG_FILE_SIZE = int(os.environ.get('LOG_FILE_SIZE', DEFAULT_LOG_FILE_SIZE))
LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', DEFAULT_LOG_BACKUP_COUNT))

# Performance logging configuration
PERFORMANCE_LOGGING = os.environ.get('PERFORMANCE_LOGGING', 'TRUE').upper() == 'TRUE'

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define log file name with timestamp to avoid overwriting
log_file = os.path.join(logs_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Define log format with colors for console output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console"""
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[41m\033[97m', # White on Red background
        'RESET': '\033[0m'    # Reset color
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

# Define formatters
file_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_formatter = ColoredFormatter(
    '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S'
)

def get_logger(name):
    """
    Returns a configured logger with the given name.
    
    Args:
        name (str): The name of the logger, typically __name__ from the calling module
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure handlers if they haven't been added yet
    if not logger.handlers:
        # Set overall logger level to the lowest of console or file level
        # to ensure messages are passed to the handlers
        logger.setLevel(min(FILE_LOG_LEVEL, CONSOLE_LOG_LEVEL))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(CONSOLE_LOG_LEVEL)
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG_FILE_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(FILE_LOG_LEVEL)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        logger.debug(f"Logger '{name}' initialized with levels: console={logging.getLevelName(CONSOLE_LOG_LEVEL)}, "
                     f"file={logging.getLevelName(FILE_LOG_LEVEL)}")
    
    return logger

# Root logger configuration - controls third-party modules
# Default to WARNING level for third-party modules to reduce noise
root_logger = logging.getLogger()
root_level = _get_log_level('ROOT_LOG_LEVEL', logging.WARNING)
root_logger.setLevel(root_level)

# --- Performance Logging Utilities ---

def log_performance(function_name=None):
    """
    Decorator to log the execution time of a function.
    
    Args:
        function_name (str, optional): Custom name for the function in logs.
            If None, uses the function's name.
            
    Returns:
        function: Decorated function with performance logging
    
    Usage:
        @log_performance()
        def my_function():
            # function code
            
        @log_performance("Custom Operation Name")
        def another_function():
            # function code
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip if performance logging is disabled
            if not PERFORMANCE_LOGGING:
                return func(*args, **kwargs)
                
            name = function_name or func.__name__
            logger = get_logger('performance')
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"PERF_METRIC - {name} completed in {execution_time:.6f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"PERF_METRIC - {name} failed after {execution_time:.6f} seconds with error: {str(e)}")
                raise
        return wrapper
    return decorator

def log_performance_metrics(metrics_dict, operation=None):
    """
    Log multiple performance metrics at once.
    
    Args:
        metrics_dict (dict): Dictionary of metric_name: value pairs
        operation (str, optional): Name of the operation these metrics belong to
    
    Usage:
        log_performance_metrics({
            'embedding_time': 0.156,
            'search_time': 0.064,
            'total_time': 0.220
        }, operation='query')
    """
    if not PERFORMANCE_LOGGING:
        return
        
    logger = get_logger('performance')
    prefix = f"{operation} - " if operation else ""
    
    for name, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            logger.info(f"PERF_METRIC - {prefix}{name}: {value:.6f}")
        else:
            logger.info(f"PERF_METRIC - {prefix}{name}: {value}")

def timeit(func=None, name=None):
    """
    Context manager to time code blocks.
    
    Args:
        func (function, optional): Function being timed (for logging)
        name (str, optional): Name to use in logs instead of function name
            
    Returns:
        float: Execution time in seconds
    
    Usage:
        # As a context manager
        with timeit(name="database_query") as timer:
            results = database.query(...)
        print(f"Query took {timer.elapsed} seconds")
    """
    class Timer:
        def __init__(self, func=None, name=None):
            self.func = func
            self.name = name
            self.start_time = None
            self.elapsed = 0
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed = time.time() - self.start_time
            
            if PERFORMANCE_LOGGING:
                logger = get_logger('performance')
                operation_name = self.name or (self.func.__name__ if self.func else "operation")
                
                if exc_type:
                    logger.error(f"PERF_METRIC - {operation_name} failed after {self.elapsed:.6f} seconds")
                else:
                    logger.info(f"PERF_METRIC - {operation_name} completed in {self.elapsed:.6f} seconds")
    
    return Timer(func, name)