"""
Prometheus metrics exporter for ML model monitoring.
"""
import os
import sys
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add src to path
sys.path.append('/app/src')

from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from monitoring.metrics import MetricsTracker
from monitoring.drift_detection import DataDriftDetector
from monitoring.data_quality import DataQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
model_accuracy = Gauge('model_accuracy', 'Model accuracy score', ['model_name', 'version'])
model_precision = Gauge('model_precision', 'Model precision score', ['model_name', 'version'])
model_recall = Gauge('model_recall', 'Model recall score', ['model_name', 'version'])
model_f1_score = Gauge('model_f1_score', 'Model F1 score', ['model_name', 'version'])

# Data drift metrics
data_drift_detected = Gauge('data_drift_detected', 'Data drift detection flag', ['feature'])
data_drift_pvalue = Gauge('data_drift_pvalue', 'Data drift p-value', ['feature'])
drift_columns_count = Gauge('drift_columns_count', 'Number of columns with drift detected')

# Data quality metrics
data_quality_score = Gauge('data_quality_score', 'Overall data quality score')
missing_values_count = Gauge('missing_values_count', 'Count of missing values', ['column'])
outliers_count = Gauge('outliers_count', 'Count of outliers detected', ['column'])
data_completeness = Gauge('data_completeness', 'Data completeness percentage', ['column'])

# API metrics
api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
prediction_latency = Histogram('prediction_latency_seconds', 'Model prediction latency')

# System metrics
model_memory_usage = Gauge('model_memory_usage_bytes', 'Model memory usage in bytes')
model_inference_time = Histogram('model_inference_time_seconds', 'Model inference time')
model_queue_size = Gauge('model_queue_size', 'Number of requests in model queue')

# Business metrics
daily_predictions = Counter('daily_predictions_total', 'Total daily predictions made')
prediction_value_distribution = Histogram('prediction_value_distribution', 'Distribution of prediction values', 
                                        buckets=[0, 100000, 200000, 300000, 500000, 750000, 1000000, float('inf')])

# Model info
model_info = Info('model_info', 'Model information')


class ModelMetricsExporter:
    """Exports ML model metrics to Prometheus."""
    
    def __init__(self, metrics_db_path: str = "/app/data/metrics.db"):
        """
        Initialize the metrics exporter.
        
        Args:
            metrics_db_path: Path to the metrics database
        """
        self.metrics_db_path = metrics_db_path
        self.metrics_tracker = None
        self.last_update = datetime.now()
        
        # Initialize metrics tracker if database exists
        if os.path.exists(metrics_db_path):
            try:
                self.metrics_tracker = MetricsTracker(storage_path=metrics_db_path)
                logger.info(f"Connected to metrics database: {metrics_db_path}")
            except Exception as e:
                logger.warning(f"Could not connect to metrics database: {e}")
        else:
            logger.warning(f"Metrics database not found: {metrics_db_path}")
    
    def update_model_performance_metrics(self):
        """Update model performance metrics from the database."""
        if not self.metrics_tracker:
            return
        
        try:
            # Get recent model metrics (last 24 hours)
            recent_metrics = self.metrics_tracker.get_recent_metrics('accuracy', hours=24)
            
            if recent_metrics:
                latest_metrics = recent_metrics[-1]  # Get most recent
                metadata = json.loads(latest_metrics.get('metadata', '{}'))
                model_name = metadata.get('model_name', 'house_price_model')
                model_version = metadata.get('model_version', 'latest')
                
                # Update Prometheus metrics
                model_accuracy.labels(model_name=model_name, version=model_version).set(latest_metrics['value'])
                
                # Get other metrics if available
                for metric_name in ['precision', 'recall', 'f1_score']:
                    metric_data = self.metrics_tracker.get_recent_metrics(metric_name, hours=1)
                    if metric_data:
                        value = metric_data[-1]['value']
                        if metric_name == 'precision':
                            model_precision.labels(model_name=model_name, version=model_version).set(value)
                        elif metric_name == 'recall':
                            model_recall.labels(model_name=model_name, version=model_version).set(value)
                        elif metric_name == 'f1_score':
                            model_f1_score.labels(model_name=model_name, version=model_version).set(value)
                
                logger.info(f"Updated model performance metrics for {model_name}:{model_version}")
            
        except Exception as e:
            logger.error(f"Error updating model performance metrics: {e}")
    
    def update_data_drift_metrics(self):
        """Update data drift metrics."""
        try:
            # Check for recent drift detection results
            if self.metrics_tracker:
                drift_metrics = self.metrics_tracker.get_recent_metrics('data_drift', hours=1)
                
                if drift_metrics:
                    latest_drift = drift_metrics[-1]
                    drift_data = json.loads(latest_drift.get('metadata', '{}'))
                    
                    # Update drift metrics
                    drift_columns_count.set(drift_data.get('drifted_columns', 0))
                    
                    # Update per-feature drift metrics
                    for feature, drift_info in drift_data.get('drift_details', {}).items():
                        data_drift_detected.labels(feature=feature).set(1 if drift_info.get('drift_detected', False) else 0)
                        if 'p_value' in drift_info:
                            data_drift_pvalue.labels(feature=feature).set(drift_info['p_value'])
                    
                    logger.info("Updated data drift metrics")
        
        except Exception as e:
            logger.error(f"Error updating data drift metrics: {e}")
    
    def update_data_quality_metrics(self):
        """Update data quality metrics."""
        try:
            if self.metrics_tracker:
                quality_metrics = self.metrics_tracker.get_recent_metrics('data_quality', hours=1)
                
                if quality_metrics:
                    latest_quality = quality_metrics[-1]
                    quality_data = json.loads(latest_quality.get('metadata', '{}'))
                    
                    # Update overall quality score
                    data_quality_score.set(quality_data.get('overall_score', 0))
                    
                    # Update column-specific quality metrics
                    for column, stats in quality_data.get('column_stats', {}).items():
                        if 'missing_count' in stats:
                            missing_values_count.labels(column=column).set(stats['missing_count'])
                        if 'completeness' in stats:
                            data_completeness.labels(column=column).set(stats['completeness'])
                        if 'outliers_count' in stats:
                            outliers_count.labels(column=column).set(stats['outliers_count'])
                    
                    logger.info("Updated data quality metrics")
        
        except Exception as e:
            logger.error(f"Error updating data quality metrics: {e}")
    
    def update_api_metrics(self):
        """Update API performance metrics."""
        try:
            if self.metrics_tracker:
                # Get API latency metrics
                latency_metrics = self.metrics_tracker.get_recent_metrics('api_latency', hours=1)
                if latency_metrics:
                    for metric in latency_metrics:
                        metadata = json.loads(metric.get('metadata', '{}'))
                        endpoint = metadata.get('endpoint', '/predict')
                        method = metadata.get('method', 'POST')
                        
                        api_request_duration.labels(method=method, endpoint=endpoint).observe(metric['value'])
                
                # Get prediction count
                prediction_metrics = self.metrics_tracker.get_recent_metrics('predictions_count', hours=24)
                if prediction_metrics:
                    total_predictions = sum(m['value'] for m in prediction_metrics)
                    daily_predictions._value._value = total_predictions  # Set counter value
                
                logger.info("Updated API metrics")
        
        except Exception as e:
            logger.error(f"Error updating API metrics: {e}")
    
    def update_model_info(self):
        """Update model information metrics."""
        try:
            model_info.info({
                'name': 'house_price_model',
                'version': 'v1.0.0',
                'framework': 'scikit-learn',
                'created_date': '2024-01-01',
                'last_updated': datetime.now().isoformat(),
                'features': 'sqft,bedrooms,bathrooms,location,year_built',
                'target': 'price',
                'model_type': 'regression'
            })
            
        except Exception as e:
            logger.error(f"Error updating model info: {e}")
    
    def run_export_cycle(self):
        """Run one complete export cycle."""
        try:
            logger.info("Starting metrics export cycle...")
            
            self.update_model_performance_metrics()
            self.update_data_drift_metrics()
            self.update_data_quality_metrics()
            self.update_api_metrics()
            self.update_model_info()
            
            self.last_update = datetime.now()
            logger.info("Metrics export cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error during export cycle: {e}")


def main():
    """Main function to start the metrics exporter."""
    # Configuration
    prometheus_port = int(os.getenv('PROMETHEUS_PORT', 8001))
    metrics_db_path = os.getenv('METRICS_DB_PATH', '/app/data/metrics.db')
    export_interval = int(os.getenv('EXPORT_INTERVAL', 30))  # seconds
    
    logger.info(f"Starting MLOps Model Metrics Exporter on port {prometheus_port}")
    logger.info(f"Metrics database path: {metrics_db_path}")
    logger.info(f"Export interval: {export_interval} seconds")
    
    # Start Prometheus HTTP server
    start_http_server(prometheus_port)
    logger.info(f"Prometheus metrics server started on port {prometheus_port}")
    
    # Initialize exporter
    exporter = ModelMetricsExporter(metrics_db_path)
    
    # Main export loop
    while True:
        try:
            exporter.run_export_cycle()
            time.sleep(export_interval)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(export_interval)


if __name__ == '__main__':
    main()
