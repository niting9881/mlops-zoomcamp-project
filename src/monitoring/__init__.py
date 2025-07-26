"""
MLOps Monitoring and Quality Assurance Module

This module provides comprehensive monitoring capabilities for MLOps pipelines including:
- Data drift detection
- Model performance monitoring
- Data quality validation
- API performance tracking
- System resource monitoring
- Quality gates and automated validation
- Monitoring dashboards and reporting

Phase 2 Features:
- Real-time monitoring and alerting
- Comprehensive data quality checks
- Model performance degradation detection
- Automated quality gates for CI/CD pipelines
- Interactive monitoring dashboards
"""

__version__ = "2.0.0"
__author__ = "MLOps Team"

# Import main monitoring components
from .drift_detection import DataDriftDetector, ModelPerformanceMonitor, SystemMetricsCollector
from .metrics import MetricsTracker, MLflowMetricsIntegration, PerformanceAlertManager
from .api_monitoring import APIMetricsCollector, monitor_api_performance, get_metrics_collector
from .data_quality import DataQualityValidator, GreatExpectationsIntegration
from .quality_gates import QualityGateEngine, QualityGateResult, GateStatus
from .dashboard import MonitoringDashboard

__all__ = [
    # Core monitoring classes
    'DataDriftDetector',
    'ModelPerformanceMonitor', 
    'SystemMetricsCollector',
    'MetricsTracker',
    'MLflowMetricsIntegration',
    'PerformanceAlertManager',
    
    # API monitoring
    'APIMetricsCollector',
    'monitor_api_performance',
    'get_metrics_collector',
    
    # Data quality
    'DataQualityValidator',
    'GreatExpectationsIntegration',
    
    # Quality gates
    'QualityGateEngine',
    'QualityGateResult',
    'GateStatus',
    
    # Dashboard
    'MonitoringDashboard'
]
