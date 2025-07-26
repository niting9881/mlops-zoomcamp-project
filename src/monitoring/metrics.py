"""
Performance metrics tracking and analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from dataclasses import dataclass, asdict
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Data class for model performance metrics."""
    timestamp: str
    model_version: str
    mae: float
    mse: float
    rmse: float
    r2: float
    sample_count: int
    prediction_mean: float
    prediction_std: float
    target_mean: float
    target_std: float
    batch_id: Optional[str] = None
    environment: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsTracker:
    """Tracks and stores model performance metrics."""
    
    def __init__(self, storage_path: str = "monitoring/metrics.db"):
        """
        Initialize metrics tracker.
        
        Args:
            storage_path: Path to SQLite database for storing metrics
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    mae REAL NOT NULL,
                    mse REAL NOT NULL,
                    rmse REAL NOT NULL,
                    r2 REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    prediction_mean REAL NOT NULL,
                    prediction_std REAL NOT NULL,
                    target_mean REAL NOT NULL,
                    target_std REAL NOT NULL,
                    batch_id TEXT,
                    environment TEXT
                )
            """)
            conn.commit()
        logger.info(f"Metrics database initialized at {self.storage_path}")
    
    def log_metrics(self, metrics: ModelMetrics) -> None:
        """
        Log model metrics to database.
        
        Args:
            metrics: ModelMetrics object to store
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT INTO model_metrics 
                (timestamp, model_version, mae, mse, rmse, r2, sample_count,
                 prediction_mean, prediction_std, target_mean, target_std, 
                 batch_id, environment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.model_version, metrics.mae, 
                metrics.mse, metrics.rmse, metrics.r2, metrics.sample_count,
                metrics.prediction_mean, metrics.prediction_std,
                metrics.target_mean, metrics.target_std,
                metrics.batch_id, metrics.environment
            ))
            conn.commit()
        
        logger.info(f"Metrics logged for model {metrics.model_version}, "
                   f"batch {metrics.batch_id}")
    
    def get_metrics_history(self, days: int = 30, 
                          model_version: str = None) -> List[ModelMetrics]:
        """
        Retrieve metrics history from database.
        
        Args:
            days: Number of days to look back
            model_version: Specific model version to filter by
            
        Returns:
            List of ModelMetrics objects
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = "SELECT * FROM model_metrics WHERE timestamp >= ?"
        params = [cutoff_date]
        
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        metrics_list = []
        for row in rows:
            metrics = ModelMetrics(
                timestamp=row['timestamp'],
                model_version=row['model_version'],
                mae=row['mae'],
                mse=row['mse'],
                rmse=row['rmse'],
                r2=row['r2'],
                sample_count=row['sample_count'],
                prediction_mean=row['prediction_mean'],
                prediction_std=row['prediction_std'],
                target_mean=row['target_mean'],
                target_std=row['target_std'],
                batch_id=row['batch_id'],
                environment=row['environment']
            )
            metrics_list.append(metrics)
        
        logger.info(f"Retrieved {len(metrics_list)} metrics records")
        return metrics_list
    
    def calculate_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate performance summary statistics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance summary dictionary
        """
        metrics_history = self.get_metrics_history(days=days)
        
        if not metrics_history:
            return {
                'period_days': days,
                'no_data': True,
                'message': f'No metrics data available for the last {days} days'
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([m.to_dict() for m in metrics_history])
        
        summary = {
            'period_days': days,
            'total_batches': len(df),
            'total_predictions': int(df['sample_count'].sum()),
            'model_versions': list(df['model_version'].unique()),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'performance_metrics': {
                'mae': {
                    'mean': float(df['mae'].mean()),
                    'std': float(df['mae'].std()),
                    'min': float(df['mae'].min()),
                    'max': float(df['mae'].max()),
                    'trend': self._calculate_trend(df['mae'].values)
                },
                'rmse': {
                    'mean': float(df['rmse'].mean()),
                    'std': float(df['rmse'].std()),
                    'min': float(df['rmse'].min()),
                    'max': float(df['rmse'].max()),
                    'trend': self._calculate_trend(df['rmse'].values)
                },
                'r2': {
                    'mean': float(df['r2'].mean()),
                    'std': float(df['r2'].std()),
                    'min': float(df['r2'].min()),
                    'max': float(df['r2'].max()),
                    'trend': self._calculate_trend(df['r2'].values)
                }
            },
            'prediction_statistics': {
                'prediction_mean': {
                    'mean': float(df['prediction_mean'].mean()),
                    'std': float(df['prediction_mean'].std())
                },
                'prediction_std': {
                    'mean': float(df['prediction_std'].mean()),
                    'std': float(df['prediction_std'].std())
                }
            }
        }
        
        return summary
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction for a metric."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.001:  # Very small slope
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'


class MLflowMetricsIntegration:
    """Integration with MLflow for metrics tracking."""
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        """
        Initialize MLflow integration.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the experiment
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient()
        logger.info("MLflow metrics integration initialized")
    
    def log_model_metrics(self, metrics: ModelMetrics, run_id: str = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: ModelMetrics object
            run_id: Specific MLflow run ID, or None for active run
        """
        metrics_dict = {
            'mae': metrics.mae,
            'mse': metrics.mse,
            'rmse': metrics.rmse,
            'r2': metrics.r2,
            'sample_count': metrics.sample_count,
            'prediction_mean': metrics.prediction_mean,
            'prediction_std': metrics.prediction_std,
            'target_mean': metrics.target_mean,
            'target_std': metrics.target_std
        }
        
        if run_id:
            for key, value in metrics_dict.items():
                self.client.log_metric(run_id, key, value)
        else:
            mlflow.log_metrics(metrics_dict)
        
        # Log additional metadata as tags
        tags = {
            'model_version': metrics.model_version,
            'batch_id': metrics.batch_id or 'unknown',
            'environment': metrics.environment or 'unknown',
            'timestamp': metrics.timestamp
        }
        
        if run_id:
            for key, value in tags.items():
                if value:
                    self.client.set_tag(run_id, key, value)
        else:
            mlflow.set_tags(tags)
        
        logger.info(f"Metrics logged to MLflow for batch {metrics.batch_id}")
    
    def get_model_performance_history(self, model_name: str, 
                                    days: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve model performance history from MLflow.
        
        Args:
            model_name: Name of the registered model
            days: Number of days to look back
            
        Returns:
            List of performance data dictionaries
        """
        try:
            # Get all versions of the model
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            performance_data = []
            cutoff_timestamp = (datetime.now() - timedelta(days=days)).timestamp() * 1000
            
            for version in model_versions:
                run_id = version.run_id
                run = self.client.get_run(run_id)
                
                # Filter by date
                if run.info.start_time < cutoff_timestamp:
                    continue
                
                metrics = run.data.metrics
                params = run.data.params
                tags = run.data.tags
                
                performance_entry = {
                    'model_version': version.version,
                    'run_id': run_id,
                    'timestamp': datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                    'metrics': metrics,
                    'params': params,
                    'tags': tags,
                    'status': version.status
                }
                
                performance_data.append(performance_entry)
            
            # Sort by timestamp
            performance_data.sort(key=lambda x: x['timestamp'], reverse=True)
            
            logger.info(f"Retrieved {len(performance_data)} performance records from MLflow")
            return performance_data
            
        except Exception as e:
            logger.error(f"Error retrieving performance history from MLflow: {str(e)}")
            return []


class PerformanceAlertManager:
    """Manages performance-based alerts and notifications."""
    
    def __init__(self, alert_config: Dict[str, Any] = None):
        """
        Initialize alert manager.
        
        Args:
            alert_config: Configuration for alert thresholds
        """
        self.alert_config = alert_config or {
            'mae_threshold': 50000,  # Alert if MAE > 50k
            'r2_threshold': 0.7,     # Alert if R² < 0.7
            'performance_drop_threshold': 0.15,  # 15% performance drop
            'consecutive_alerts': 3   # Number of consecutive bad batches to trigger alert
        }
        self.alert_history = []
        logger.info("Performance alert manager initialized")
    
    def check_performance_alerts(self, current_metrics: ModelMetrics,
                                baseline_metrics: ModelMetrics = None) -> List[Dict[str, Any]]:
        """
        Check for performance alerts.
        
        Args:
            current_metrics: Current batch metrics
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Absolute threshold alerts
        if current_metrics.mae > self.alert_config['mae_threshold']:
            alerts.append({
                'type': 'high_mae',
                'severity': 'medium',
                'message': f"MAE {current_metrics.mae:.2f} exceeds threshold {self.alert_config['mae_threshold']}",
                'current_value': current_metrics.mae,
                'threshold': self.alert_config['mae_threshold'],
                'timestamp': current_metrics.timestamp
            })
        
        if current_metrics.r2 < self.alert_config['r2_threshold']:
            alerts.append({
                'type': 'low_r2',
                'severity': 'medium',
                'message': f"R² {current_metrics.r2:.4f} below threshold {self.alert_config['r2_threshold']}",
                'current_value': current_metrics.r2,
                'threshold': self.alert_config['r2_threshold'],
                'timestamp': current_metrics.timestamp
            })
        
        # Comparative alerts (if baseline provided)
        if baseline_metrics:
            mae_change = (current_metrics.mae - baseline_metrics.mae) / baseline_metrics.mae
            r2_change = (baseline_metrics.r2 - current_metrics.r2) / baseline_metrics.r2
            
            if mae_change > self.alert_config['performance_drop_threshold']:
                alerts.append({
                    'type': 'mae_degradation',
                    'severity': 'high',
                    'message': f"MAE increased by {mae_change*100:.1f}% compared to baseline",
                    'current_value': current_metrics.mae,
                    'baseline_value': baseline_metrics.mae,
                    'change_percent': mae_change * 100,
                    'timestamp': current_metrics.timestamp
                })
            
            if r2_change > self.alert_config['performance_drop_threshold']:
                alerts.append({
                    'type': 'r2_degradation',
                    'severity': 'high',
                    'message': f"R² decreased by {r2_change*100:.1f}% compared to baseline",
                    'current_value': current_metrics.r2,
                    'baseline_value': baseline_metrics.r2,
                    'change_percent': r2_change * 100,
                    'timestamp': current_metrics.timestamp
                })
        
        # Store alerts in history
        for alert in alerts:
            self.alert_history.append(alert)
        
        if alerts:
            logger.warning(f"Generated {len(alerts)} performance alerts")
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts within specified time window.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = []
        for alert in self.alert_history:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time >= cutoff_time:
                recent_alerts.append(alert)
        
        return recent_alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alert statistics.
        
        Returns:
            Alert summary dictionary
        """
        if not self.alert_history:
            return {'total_alerts': 0, 'message': 'No alerts recorded'}
        
        # Count alerts by type and severity
        alert_counts = {}
        severity_counts = {}
        
        for alert in self.alert_history:
            alert_type = alert['type']
            severity = alert['severity']
            
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        recent_24h = len(self.get_recent_alerts(hours=24))
        recent_1h = len(self.get_recent_alerts(hours=1))
        
        return {
            'total_alerts': len(self.alert_history),
            'alerts_by_type': alert_counts,
            'alerts_by_severity': severity_counts,
            'recent_24h': recent_24h,
            'recent_1h': recent_1h,
            'latest_alert': self.alert_history[-1] if self.alert_history else None
        }


def create_performance_dashboard_data(metrics_tracker: MetricsTracker,
                                    days: int = 7) -> Dict[str, Any]:
    """
    Create data structure for performance monitoring dashboard.
    
    Args:
        metrics_tracker: MetricsTracker instance
        days: Number of days to analyze
        
    Returns:
        Dashboard data dictionary
    """
    # Get performance summary
    summary = metrics_tracker.calculate_performance_summary(days=days)
    
    # Get detailed metrics history
    metrics_history = metrics_tracker.get_metrics_history(days=days)
    
    # Prepare time series data
    time_series_data = []
    for metrics in metrics_history:
        time_series_data.append({
            'timestamp': metrics.timestamp,
            'mae': metrics.mae,
            'rmse': metrics.rmse,
            'r2': metrics.r2,
            'sample_count': metrics.sample_count,
            'model_version': metrics.model_version
        })
    
    # Calculate daily aggregates
    if metrics_history:
        df = pd.DataFrame([m.to_dict() for m in metrics_history])
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        daily_aggregates = df.groupby('date').agg({
            'mae': ['mean', 'std', 'count'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'sample_count': 'sum'
        }).round(4)
        
        daily_data = []
        for date in daily_aggregates.index:
            daily_data.append({
                'date': str(date),
                'mae_mean': daily_aggregates.loc[date, ('mae', 'mean')],
                'mae_std': daily_aggregates.loc[date, ('mae', 'std')],
                'rmse_mean': daily_aggregates.loc[date, ('rmse', 'mean')],
                'r2_mean': daily_aggregates.loc[date, ('r2', 'mean')],
                'batch_count': int(daily_aggregates.loc[date, ('mae', 'count')]),
                'total_predictions': int(daily_aggregates.loc[date, ('sample_count', 'sum')])
            })
    else:
        daily_data = []
    
    return {
        'summary': summary,
        'time_series': time_series_data,
        'daily_aggregates': daily_data,
        'generated_at': datetime.now().isoformat()
    }
