from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import sys
import os
import logging
from datetime import datetime
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config_manager import get_api_config, get_logging_config
from inference import predict_price, batch_predict
from schemas import HousePredictionRequest, PredictionResponse
from utils import health_check, handle_errors, log_execution_time, setup_logging

# Setup logging
logging_config = get_logging_config()
logger = setup_logging(
    log_level=logging_config.level,
    log_file=logging_config.file_path
)

# Get API configuration
api_config = get_api_config()

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request latency', ['method', 'endpoint'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Model prediction latency')
PREDICTIONS_COUNT = Counter('daily_predictions_total', 'Total predictions made')
ACTIVE_REQUESTS = Gauge('api_active_requests', 'Number of active API requests')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load the model')

# Initialize FastAPI app with metadata
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "An API for predicting house prices based on various features. "
        "This application is part of the MLOps Bootcamp by School of Devops. "
        "Authored by Gourav Shah."
    ),
    version="1.0.0",
    contact={
        "name": "School of Devops",
        "url": "https://schoolofdevops.com",
        "email": "learn@schoolofdevops.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics for all API requests."""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
    
    finally:
        ACTIVE_REQUESTS.dec()

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception for {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions."""
    logger.warning(f"HTTP exception for {request.url}: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.2f}s")
    
    return response

# Health check endpoint
@app.get("/health", response_model=dict)
@log_execution_time
async def health_check_endpoint():
    """
    Health check endpoint with comprehensive system status.
    
    Returns:
        Dict containing health status and system information
    """
    try:
        health_status = health_check()
        logger.info("Health check completed successfully")
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Prometheus metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus metrics in text format
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
@log_execution_time
@handle_errors
async def predict(request: HousePredictionRequest):
    """
    Predict house price based on input features.
    
    Args:
        request: House prediction request with features
    
    Returns:
        Prediction response with price estimate and confidence interval
    """
    try:
        logger.info(f"Prediction request received: {request.dict()}")
        
        # Track prediction latency
        prediction_start = time.time()
        
        # Make prediction
        prediction = predict_price(request)
        
        # Record prediction metrics
        prediction_duration = time.time() - prediction_start
        PREDICTION_LATENCY.observe(prediction_duration)
        PREDICTIONS_COUNT.inc()
        
        logger.info(f"Prediction completed: ${prediction.predicted_price:,.2f}")
        return prediction
        
    except ValueError as e:
        logger.error(f"Validation error in prediction: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction service error")

# Batch prediction endpoint
@app.post("/batch-predict", response_model=list)
@log_execution_time
@handle_errors
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    """
    Perform batch predictions for multiple houses.
    
    Args:
        requests: List of house prediction requests
    
    Returns:
        List of prediction values
    """
    try:
        if not requests:
            logger.warning("Empty batch prediction request received")
            return []
        
        logger.info(f"Batch prediction request received with {len(requests)} items")
        
        # Make batch predictions
        predictions = batch_predict(requests)
        
        logger.info(f"Batch prediction completed for {len(predictions)} items")
        return predictions
        
    except ValueError as e:
        logger.error(f"Validation error in batch prediction: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch prediction service error")

# Model information endpoint
@app.get("/model/info", response_model=dict)
@log_execution_time
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary containing model metadata
    """
    try:
        from utils import get_model_info
        info = get_model_info()
        logger.info("Model info requested")
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Unable to retrieve model information")

# API status endpoint
@app.get("/status", response_model=dict)
async def api_status():
    """
    Get API status and configuration information.
    
    Returns:
        Dictionary containing API status
    """
    return {
        "status": "running",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.now().isoformat(),
        "uptime": "See logs for startup time",  # Could implement actual uptime tracking
        "endpoints": [
            "/health",
            "/predict",
            "/batch-predict",
            "/model/info",
            "/status"
        ]
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting House Price Prediction API")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"API running on {api_config.host}:{api_config.port}")
    
    # Perform startup health checks
    try:
        health_status = health_check()
        if health_status["status"] != "healthy":
            logger.warning("Application started but health check indicates issues")
        else:
            logger.info("Application started successfully - all health checks passed")
    except Exception as e:
        logger.error(f"Startup health check failed: {str(e)}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down House Price Prediction API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.reload,
        workers=api_config.workers,
        log_level=logging_config.level.lower()
    )