#!/usr/bin/env python3
"""
Simplified Model Metrics Exporter for Prometheus/Grafana
"""
import time
import random
import sqlite3
import logging
from pathlib import Path
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client.core import CollectorRegistry, REGISTRY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModelMetricsExporter:
    """Simple metrics exporter for ML model monitoring"""
    
    def __init__(self, port=8001):
        self.port = port
        
        # Create metrics registry
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.model_predictions_total = Counter(
            'model_predictions_total',
            'Total number of model predictions',
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy score',
            registry=self.registry
        )
        
        self.model_rmse = Gauge(
            'model_rmse',
            'Model RMSE score',
            registry=self.registry
        )
        
        self.model_r2_score = Gauge(
            'model_r2_score',
            'Model R² score',
            registry=self.registry
        )
        
        self.data_drift_detected = Gauge(
            'data_drift_detected',
            'Data drift detection status (1=detected, 0=not detected)',
            ['feature']
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality validation score',
            registry=self.registry
        )
        
        self.api_response_time = Histogram(
            'api_response_time_seconds',
            'API response time in seconds',
            registry=self.registry
        )
        
        self.drift_p_value = Gauge(
            'drift_p_value',
            'Statistical drift test p-value',
            ['feature'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
    def generate_sample_metrics(self):
        """Generate sample metrics for demonstration"""
        # Model performance metrics
        self.model_accuracy.set(random.uniform(0.85, 0.95))
        self.model_rmse.set(random.uniform(0.1, 0.3))
        self.model_r2_score.set(random.uniform(0.80, 0.95))
        
        # Data quality
        self.data_quality_score.set(random.uniform(0.90, 1.0))
        
        # API metrics
        self.api_response_time.observe(random.uniform(0.1, 0.5))
        
        # System metrics
        self.system_cpu_usage.set(random.uniform(10, 80))
        self.system_memory_usage.set(random.uniform(40, 90))
        
        # Feature drift metrics
        features = ['bedrooms', 'bathrooms', 'sqft_living']
        for feature in features:
            drift_status = random.choice([0, 1])  # 0 = no drift, 1 = drift detected
            p_value = random.uniform(0.01, 0.1) if drift_status else random.uniform(0.1, 1.0)
            
            self.data_drift_detected.labels(feature=feature).set(drift_status)
            self.drift_p_value.labels(feature=feature).set(p_value)
        
        # Increment prediction counter
        self.model_predictions_total.inc(random.randint(1, 10))
        
        logger.info("Generated sample metrics")
        
    def start_server(self):
        """Start the metrics server"""
        try:
            # Start HTTP server for metrics endpoint
            start_http_server(self.port, registry=self.registry)
            logger.info(f"Metrics server started on port {self.port}")
            
            # Generate metrics every 30 seconds
            while True:
                self.generate_sample_metrics()
                time.sleep(30)
                
        except Exception as e:
            logger.error(f"Error starting metrics server: {e}")
            raise

def main():
    """Main function"""
    logger.info("Starting Simple Model Metrics Exporter...")
    
    exporter = SimpleModelMetricsExporter()
    exporter.start_server()

if __name__ == "__main__":
    main()
