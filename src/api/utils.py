"""
Utility functions for the House Price Prediction API.
"""
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json
import traceback
from functools import wraps
import pandas as pd
import numpy as np

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Returns:
        Logger instance
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE")
)

def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"Starting execution of {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

def handle_errors(func):
    """Decorator for comprehensive error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_details = {
                "function": func.__name__,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            logger.error(f"Error in {func.__name__}: {json.dumps(error_details, indent=2)}")
            raise
    
    return wrapper

def validate_model_files(model_path: str, preprocessor_path: str) -> bool:
    """
    Validate that required model files exist and are accessible.
    
    Args:
        model_path: Path to the trained model file
        preprocessor_path: Path to the preprocessor file
    
    Returns:
        True if all files are valid, False otherwise
    """
    try:
        model_file = Path(model_path)
        preprocessor_file = Path(preprocessor_path)
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        if not preprocessor_file.exists():
            logger.error(f"Preprocessor file not found: {preprocessor_path}")
            return False
        
        # Check file sizes (should be > 0)
        if model_file.stat().st_size == 0:
            logger.error(f"Model file is empty: {model_path}")
            return False
        
        if preprocessor_file.stat().st_size == 0:
            logger.error(f"Preprocessor file is empty: {preprocessor_path}")
            return False
        
        logger.info(f"Model files validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating model files: {str(e)}")
        return False

def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize input data.
    
    Args:
        data: Input data dictionary
    
    Returns:
        Validated and sanitized data dictionary
    """
    errors = []
    validated_data = data.copy()
    
    # Required fields
    required_fields = ['sqft', 'bedrooms', 'bathrooms', 'location', 'year_built', 'condition']
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate numeric fields
    if 'sqft' in data:
        try:
            sqft = float(data['sqft'])
            if sqft <= 0:
                errors.append("sqft must be positive")
            elif sqft > 50000:  # Reasonable upper limit
                errors.append("sqft seems unreasonably large (>50,000)")
            validated_data['sqft'] = sqft
        except (ValueError, TypeError):
            errors.append("sqft must be a valid number")
    
    if 'bedrooms' in data:
        try:
            bedrooms = int(data['bedrooms'])
            if bedrooms < 1 or bedrooms > 20:
                errors.append("bedrooms must be between 1 and 20")
            validated_data['bedrooms'] = bedrooms
        except (ValueError, TypeError):
            errors.append("bedrooms must be a valid integer")
    
    if 'bathrooms' in data:
        try:
            bathrooms = float(data['bathrooms'])
            if bathrooms <= 0 or bathrooms > 20:
                errors.append("bathrooms must be between 0 and 20")
            validated_data['bathrooms'] = bathrooms
        except (ValueError, TypeError):
            errors.append("bathrooms must be a valid number")
    
    if 'year_built' in data:
        try:
            year_built = int(data['year_built'])
            current_year = datetime.now().year
            if year_built < 1800 or year_built > current_year:
                errors.append(f"year_built must be between 1800 and {current_year}")
            validated_data['year_built'] = year_built
        except (ValueError, TypeError):
            errors.append("year_built must be a valid integer")
    
    # Validate categorical fields
    if 'location' in data:
        valid_locations = ['urban', 'suburban', 'rural']
        if data['location'].lower() not in valid_locations:
            errors.append(f"location must be one of: {', '.join(valid_locations)}")
        validated_data['location'] = data['location'].lower()
    
    if 'condition' in data:
        valid_conditions = ['poor', 'fair', 'good', 'excellent']
        if data['condition'].lower() not in valid_conditions:
            errors.append(f"condition must be one of: {', '.join(valid_conditions)}")
        validated_data['condition'] = data['condition'].lower()
    
    if errors:
        error_msg = "; ".join(errors)
        logger.error(f"Input validation failed: {error_msg}")
        raise ValueError(f"Input validation failed: {error_msg}")
    
    logger.info("Input data validated successfully")
    return validated_data

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary containing model metadata
    """
    return {
        "model_version": "1.0.0",
        "model_type": "XGBoost Regressor",
        "features": ["sqft", "bedrooms", "bathrooms", "location", "year_built", "condition"],
        "target": "price",
        "last_trained": "2024-01-01",  # This should come from model metadata
        "performance_metrics": {
            "mae": 16370.50,  # This should come from model metadata
            "r2_score": 0.85   # This should come from model metadata
        }
    }

def sanitize_prediction_output(prediction: float, confidence_interval: list) -> Dict[str, Any]:
    """
    Sanitize and format prediction output.
    
    Args:
        prediction: Raw prediction value
        confidence_interval: Raw confidence interval
    
    Returns:
        Sanitized prediction output
    """
    try:
        # Handle numpy types and ensure proper formatting
        sanitized_prediction = float(prediction)
        sanitized_ci = [float(ci) for ci in confidence_interval]
        
        # Round to reasonable precision
        sanitized_prediction = round(sanitized_prediction, 2)
        sanitized_ci = [round(ci, 2) for ci in sanitized_ci]
        
        # Validate prediction is reasonable
        if sanitized_prediction < 0:
            logger.warning(f"Negative prediction detected: {sanitized_prediction}")
            sanitized_prediction = 0.0
        
        if sanitized_prediction > 10000000:  # 10M seems like a reasonable upper limit
            logger.warning(f"Extremely high prediction detected: {sanitized_prediction}")
        
        return {
            "predicted_price": sanitized_prediction,
            "confidence_interval": sanitized_ci,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error sanitizing prediction output: {str(e)}")
        raise ValueError(f"Invalid prediction output: {str(e)}")

def health_check() -> Dict[str, Any]:
    """
    Perform comprehensive health check.
    
    Returns:
        Health status dictionary
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    try:
        # Check model files
        model_path = "models/trained/house_price_model.pkl"
        preprocessor_path = "models/trained/preprocessor.pkl"
        
        health_status["checks"]["model_files"] = validate_model_files(model_path, preprocessor_path)
        
        # Check memory usage
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        health_status["checks"]["memory_usage_mb"] = round(memory_info.rss / 1024 / 1024, 2)
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        health_status["checks"]["disk_usage_percent"] = round((disk_usage.used / disk_usage.total) * 100, 2)
        
        # Overall status
        if not health_status["checks"]["model_files"]:
            health_status["status"] = "unhealthy"
        
        logger.info(f"Health check completed: {health_status['status']}")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        return health_status