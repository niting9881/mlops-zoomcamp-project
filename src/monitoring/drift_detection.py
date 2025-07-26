"""
Data drift detection and monitoring module.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detects data drift between reference and current datasets."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            threshold: P-value threshold for statistical tests
        """
        self.threshold = threshold
        self.reference_stats = {}
        
    def fit_reference(self, reference_data: pd.DataFrame) -> None:
        """
        Fit reference dataset statistics.
        
        Args:
            reference_data: Reference dataset to establish baseline
        """
        logger.info("Computing reference dataset statistics...")
        
        for column in reference_data.columns:
            if reference_data[column].dtype in ['int64', 'float64']:
                # Numerical column statistics
                self.reference_stats[column] = {
                    'type': 'numerical',
                    'mean': float(reference_data[column].mean()),
                    'std': float(reference_data[column].std()),
                    'min': float(reference_data[column].min()),
                    'max': float(reference_data[column].max()),
                    'q25': float(reference_data[column].quantile(0.25)),
                    'q50': float(reference_data[column].quantile(0.50)),
                    'q75': float(reference_data[column].quantile(0.75)),
                    'sample': reference_data[column].dropna().values.tolist()[:1000]  # Sample for KS test
                }
            else:
                # Categorical column statistics
                value_counts = reference_data[column].value_counts(normalize=True)
                self.reference_stats[column] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict(),
                    'unique_count': len(value_counts),
                    'top_categories': value_counts.head(10).to_dict()
                }
        
        logger.info(f"Reference statistics computed for {len(self.reference_stats)} columns")
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare against reference
            
        Returns:
            Dictionary containing drift detection results
        """
        if not self.reference_stats:
            raise ValueError("Reference statistics not fitted. Call fit_reference() first.")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_detected': False,
            'columns_with_drift': [],
            'drift_details': {},
            'summary': {
                'total_columns': 0,
                'drifted_columns': 0,
                'drift_percentage': 0.0
            }
        }
        
        logger.info("Detecting data drift...")
        
        for column in self.reference_stats.keys():
            if column not in current_data.columns:
                logger.warning(f"Column {column} not found in current data")
                continue
                
            column_stats = self.reference_stats[column]
            drift_results['drift_details'][column] = self._detect_column_drift(
                column, current_data[column], column_stats
            )
            
            if drift_results['drift_details'][column]['drift_detected']:
                drift_results['columns_with_drift'].append(column)
                drift_results['overall_drift_detected'] = True
        
        # Summary statistics
        drift_results['summary']['total_columns'] = len(self.reference_stats)
        drift_results['summary']['drifted_columns'] = len(drift_results['columns_with_drift'])
        drift_results['summary']['drift_percentage'] = (
            drift_results['summary']['drifted_columns'] / 
            drift_results['summary']['total_columns'] * 100
        )
        
        logger.info(f"Drift detection complete. {drift_results['summary']['drifted_columns']} "
                   f"out of {drift_results['summary']['total_columns']} columns show drift")
        
        return drift_results
    
    def _detect_column_drift(self, column_name: str, current_data: pd.Series, 
                           reference_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Detect drift for a single column."""
        result = {
            'drift_detected': False,
            'p_value': None,
            'test_used': None,
            'magnitude': None,
            'details': {}
        }
        
        try:
            if reference_stats['type'] == 'numerical':
                # Kolmogorov-Smirnov test for numerical data
                reference_sample = np.array(reference_stats['sample'])
                current_sample = current_data.dropna().values
                
                if len(current_sample) == 0:
                    logger.warning(f"No valid data for column {column_name}")
                    return result
                
                # Limit sample size for computational efficiency
                if len(current_sample) > 1000:
                    current_sample = np.random.choice(current_sample, 1000, replace=False)
                
                ks_statistic, p_value = stats.ks_2samp(reference_sample, current_sample)
                
                result.update({
                    'p_value': float(p_value),
                    'test_used': 'kolmogorov_smirnov',
                    'magnitude': float(ks_statistic),
                    'drift_detected': p_value < self.threshold,
                    'details': {
                        'ks_statistic': float(ks_statistic),
                        'current_mean': float(current_data.mean()),
                        'reference_mean': reference_stats['mean'],
                        'current_std': float(current_data.std()),
                        'reference_std': reference_stats['std']
                    }
                })
                
            else:
                # Chi-square test for categorical data
                current_dist = current_data.value_counts(normalize=True)
                reference_dist = reference_stats['distribution']
                
                # Align distributions
                all_categories = set(current_dist.index) | set(reference_dist.keys())
                current_aligned = [current_dist.get(cat, 0) for cat in all_categories]
                reference_aligned = [reference_dist.get(cat, 0) for cat in all_categories]
                
                # Add small epsilon to avoid zero frequencies
                epsilon = 1e-6
                current_aligned = [max(val, epsilon) for val in current_aligned]
                reference_aligned = [max(val, epsilon) for val in reference_aligned]
                
                chi2_stat, p_value = stats.chisquare(current_aligned, reference_aligned)
                
                result.update({
                    'p_value': float(p_value),
                    'test_used': 'chi_square',
                    'magnitude': float(chi2_stat),
                    'drift_detected': p_value < self.threshold,
                    'details': {
                        'chi2_statistic': float(chi2_stat),
                        'current_unique_count': len(current_dist),
                        'reference_unique_count': reference_stats['unique_count'],
                        'new_categories': list(set(current_dist.index) - set(reference_dist.keys())),
                        'missing_categories': list(set(reference_dist.keys()) - set(current_dist.index))
                    }
                })
                
        except Exception as e:
            logger.error(f"Error detecting drift for column {column_name}: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def save_reference_stats(self, filepath: str) -> None:
        """Save reference statistics to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.reference_stats, f, indent=2)
        logger.info(f"Reference statistics saved to {filepath}")
    
    def load_reference_stats(self, filepath: str) -> None:
        """Load reference statistics from file."""
        with open(filepath, 'r') as f:
            self.reference_stats = json.load(f)
        logger.info(f"Reference statistics loaded from {filepath}")


class ModelPerformanceMonitor:
    """Monitors model performance over time."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize performance monitor.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.performance_history = []
        
        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
    
    def evaluate_batch(self, X: pd.DataFrame, y_true: pd.Series, 
                      batch_id: str = None) -> Dict[str, Any]:
        """
        Evaluate model performance on a batch of data.
        
        Args:
            X: Features
            y_true: True targets
            batch_id: Identifier for this batch
            
        Returns:
            Performance metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not loaded. Provide model_path or call load_model()")
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = {
            'batch_id': batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(X),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'predictions_mean': float(np.mean(y_pred)),
            'predictions_std': float(np.std(y_pred)),
            'targets_mean': float(np.mean(y_true)),
            'targets_std': float(np.std(y_true))
        }
        
        # Store in history
        self.performance_history.append(metrics)
        
        logger.info(f"Batch {metrics['batch_id']} evaluated: "
                   f"MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}")
        
        return metrics
    
    def detect_performance_degradation(self, baseline_metrics: Dict[str, float] = None,
                                     degradation_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect if model performance has degraded significantly.
        
        Args:
            baseline_metrics: Baseline performance metrics
            degradation_threshold: Threshold for performance degradation (0.1 = 10%)
            
        Returns:
            Degradation detection results
        """
        if not self.performance_history:
            return {'degradation_detected': False, 'message': 'No performance history available'}
        
        # Use baseline or first recorded performance
        if baseline_metrics is None:
            baseline_metrics = self.performance_history[0]
        
        current_metrics = self.performance_history[-1]
        
        # Calculate degradation for key metrics
        degradation_results = {
            'degradation_detected': False,
            'degraded_metrics': [],
            'baseline_timestamp': baseline_metrics.get('timestamp', 'unknown'),
            'current_timestamp': current_metrics['timestamp'],
            'degradation_details': {}
        }
        
        key_metrics = ['mae', 'rmse', 'r2']
        
        for metric in key_metrics:
            if metric in baseline_metrics and metric in current_metrics:
                baseline_val = baseline_metrics[metric]
                current_val = current_metrics[metric]
                
                # For MAE and RMSE, higher is worse; for R², lower is worse
                if metric in ['mae', 'rmse']:
                    degradation = (current_val - baseline_val) / baseline_val
                else:  # R²
                    degradation = (baseline_val - current_val) / baseline_val
                
                degradation_results['degradation_details'][metric] = {
                    'baseline_value': baseline_val,
                    'current_value': current_val,
                    'degradation_percentage': degradation * 100,
                    'is_degraded': degradation > degradation_threshold
                }
                
                if degradation > degradation_threshold:
                    degradation_results['degraded_metrics'].append(metric)
                    degradation_results['degradation_detected'] = True
        
        return degradation_results
    
    def get_performance_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            window_size: Number of recent batches to analyze
            
        Returns:
            Trend analysis results
        """
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data', 'message': 'Need at least 2 data points'}
        
        recent_history = self.performance_history[-window_size:]
        
        # Calculate trends
        trends = {}
        for metric in ['mae', 'rmse', 'r2']:
            values = [batch[metric] for batch in recent_history if metric in batch]
            if len(values) >= 2:
                # Simple linear trend
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                trends[metric] = {
                    'slope': float(slope),
                    'direction': 'improving' if (
                        slope < 0 and metric in ['mae', 'rmse'] or 
                        slope > 0 and metric == 'r2'
                    ) else 'degrading',
                    'recent_values': values[-5:],  # Last 5 values
                    'trend_strength': abs(slope)
                }
        
        return {
            'window_size': len(recent_history),
            'trends': trends,
            'overall_trend': self._determine_overall_trend(trends)
        }
    
    def _determine_overall_trend(self, trends: Dict[str, Dict]) -> str:
        """Determine overall performance trend."""
        improving_count = sum(1 for t in trends.values() if t['direction'] == 'improving')
        degrading_count = sum(1 for t in trends.values() if t['direction'] == 'degrading')
        
        if improving_count > degrading_count:
            return 'improving'
        elif degrading_count > improving_count:
            return 'degrading'
        else:
            return 'stable'
    
    def save_performance_history(self, filepath: str) -> None:
        """Save performance history to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        logger.info(f"Performance history saved to {filepath}")
    
    def load_performance_history(self, filepath: str) -> None:
        """Load performance history from file."""
        with open(filepath, 'r') as f:
            self.performance_history = json.load(f)
        logger.info(f"Performance history loaded from {filepath}")


class SystemMetricsCollector:
    """Collects system-level metrics for monitoring."""
    
    @staticmethod
    def collect_metrics() -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'usage_percent': memory.percent
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'usage_percent': round((disk.used / disk.total) * 100, 2)
                }
            }
            
            return metrics
            
        except ImportError:
            logger.warning("psutil not available for system metrics collection")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': 'psutil_not_available'
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


def generate_monitoring_report(drift_results: Dict[str, Any] = None,
                             performance_results: Dict[str, Any] = None,
                             system_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate comprehensive monitoring report.
    
    Args:
        drift_results: Data drift detection results
        performance_results: Model performance results
        system_metrics: System metrics
        
    Returns:
        Comprehensive monitoring report
    """
    report = {
        'report_timestamp': datetime.now().isoformat(),
        'summary': {
            'status': 'healthy',
            'alerts': []
        },
        'data_drift': drift_results,
        'model_performance': performance_results,
        'system_metrics': system_metrics
    }
    
    # Determine overall health status
    alerts = []
    
    if drift_results and drift_results.get('overall_drift_detected'):
        alerts.append({
            'type': 'data_drift',
            'severity': 'medium',
            'message': f"Data drift detected in {len(drift_results['columns_with_drift'])} columns",
            'details': drift_results['columns_with_drift']
        })
    
    if performance_results and performance_results.get('degradation_detected'):
        alerts.append({
            'type': 'performance_degradation',
            'severity': 'high',
            'message': f"Performance degradation detected in: {', '.join(performance_results['degraded_metrics'])}",
            'details': performance_results['degradation_details']
        })
    
    if system_metrics and 'error' not in system_metrics:
        # Check for system resource alerts
        if system_metrics.get('memory', {}).get('usage_percent', 0) > 85:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'medium',
                'message': f"High memory usage: {system_metrics['memory']['usage_percent']:.1f}%"
            })
        
        if system_metrics.get('disk', {}).get('usage_percent', 0) > 90:
            alerts.append({
                'type': 'high_disk_usage',
                'severity': 'high',
                'message': f"High disk usage: {system_metrics['disk']['usage_percent']:.1f}%"
            })
    
    report['summary']['alerts'] = alerts
    report['summary']['status'] = 'critical' if any(a['severity'] == 'high' for a in alerts) else \
                                  'warning' if alerts else 'healthy'
    
    logger.info(f"Monitoring report generated with status: {report['summary']['status']}")
    
    return report
