"""
API monitoring and observability module.
"""
import logging
import time
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path
from collections import deque
import threading
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class APIMetrics:
    """Data class for API performance metrics."""
    timestamp: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    error_message: Optional[str] = None


class APIMetricsCollector:
    """Collects and stores API performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        """
        Initialize API metrics collector.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.metrics = deque(maxlen=max_metrics)
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        
        # Real-time statistics
        self.total_requests = 0
        self.error_count = 0
        self.endpoint_stats = {}
        
        logger.info("API metrics collector initialized")
    
    def record_request(self, metrics: APIMetrics) -> None:
        """
        Record API request metrics.
        
        Args:
            metrics: APIMetrics object to record
        """
        with self.lock:
            self.metrics.append(metrics)
            self.total_requests += 1
            
            # Update error count
            if metrics.status_code >= 400:
                self.error_count += 1
            
            # Update endpoint statistics
            endpoint_key = f"{metrics.method} {metrics.endpoint}"
            if endpoint_key not in self.endpoint_stats:
                self.endpoint_stats[endpoint_key] = {
                    'count': 0,
                    'total_response_time': 0,
                    'error_count': 0,
                    'min_response_time': float('inf'),
                    'max_response_time': 0
                }
            
            stats = self.endpoint_stats[endpoint_key]
            stats['count'] += 1
            stats['total_response_time'] += metrics.response_time_ms
            
            if metrics.status_code >= 400:
                stats['error_count'] += 1
            
            stats['min_response_time'] = min(stats['min_response_time'], metrics.response_time_ms)
            stats['max_response_time'] = max(stats['max_response_time'], metrics.response_time_ms)
    
    def get_metrics_summary(self, minutes: int = 15) -> Dict[str, Any]:
        """
        Get summary of API metrics for the specified time window.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            Metrics summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            # Filter metrics by time window
            recent_metrics = [
                m for m in self.metrics 
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]
            
            if not recent_metrics:
                return {
                    'time_window_minutes': minutes,
                    'no_data': True,
                    'message': f'No API metrics in the last {minutes} minutes'
                }
            
            # Calculate summary statistics
            total_requests = len(recent_metrics)
            error_requests = sum(1 for m in recent_metrics if m.status_code >= 400)
            response_times = [m.response_time_ms for m in recent_metrics]
            
            # Calculate percentiles
            response_times.sort()
            p50_idx = int(len(response_times) * 0.5)
            p95_idx = int(len(response_times) * 0.95)
            p99_idx = int(len(response_times) * 0.99)
            
            # Group by endpoint
            endpoint_breakdown = {}
            for metric in recent_metrics:
                endpoint = f"{metric.method} {metric.endpoint}"
                if endpoint not in endpoint_breakdown:
                    endpoint_breakdown[endpoint] = {
                        'count': 0,
                        'error_count': 0,
                        'avg_response_time': 0,
                        'total_response_time': 0
                    }
                
                breakdown = endpoint_breakdown[endpoint]
                breakdown['count'] += 1
                breakdown['total_response_time'] += metric.response_time_ms
                
                if metric.status_code >= 400:
                    breakdown['error_count'] += 1
            
            # Calculate averages
            for endpoint, stats in endpoint_breakdown.items():
                stats['avg_response_time'] = stats['total_response_time'] / stats['count']
                stats['error_rate'] = (stats['error_count'] / stats['count']) * 100
                del stats['total_response_time']  # Remove intermediate calculation
            
            summary = {
                'time_window_minutes': minutes,
                'total_requests': total_requests,
                'error_requests': error_requests,
                'error_rate_percent': (error_requests / total_requests) * 100 if total_requests > 0 else 0,
                'requests_per_minute': total_requests / minutes,
                'response_time_stats': {
                    'min_ms': min(response_times),
                    'max_ms': max(response_times),
                    'mean_ms': sum(response_times) / len(response_times),
                    'p50_ms': response_times[p50_idx] if p50_idx < len(response_times) else 0,
                    'p95_ms': response_times[p95_idx] if p95_idx < len(response_times) else 0,
                    'p99_ms': response_times[p99_idx] if p99_idx < len(response_times) else 0
                },
                'endpoint_breakdown': endpoint_breakdown,
                'time_range': {
                    'start': cutoff_time.isoformat(),
                    'end': datetime.now().isoformat()
                }
            }
            
            return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status based on API metrics.
        
        Returns:
            Health status dictionary
        """
        # Get recent metrics (last 5 minutes)
        recent_summary = self.get_metrics_summary(minutes=5)
        
        if recent_summary.get('no_data'):
            return {
                'status': 'unknown',
                'message': 'No recent API activity to assess health'
            }
        
        # Determine health based on error rate and response times
        error_rate = recent_summary['error_rate_percent']
        avg_response_time = recent_summary['response_time_stats']['mean_ms']
        p95_response_time = recent_summary['response_time_stats']['p95_ms']
        
        alerts = []
        
        # Check error rate
        if error_rate > 20:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f'Error rate is {error_rate:.1f}% (threshold: 20%)'
            })
        elif error_rate > 10:
            alerts.append({
                'type': 'elevated_error_rate',
                'severity': 'warning',
                'message': f'Error rate is {error_rate:.1f}% (threshold: 10%)'
            })
        
        # Check response times
        if p95_response_time > 5000:  # 5 seconds
            alerts.append({
                'type': 'slow_response_time',
                'severity': 'critical',
                'message': f'95th percentile response time is {p95_response_time:.0f}ms (threshold: 5000ms)'
            })
        elif avg_response_time > 2000:  # 2 seconds
            alerts.append({
                'type': 'elevated_response_time',
                'severity': 'warning',
                'message': f'Average response time is {avg_response_time:.0f}ms (threshold: 2000ms)'
            })
        
        # Determine overall status
        if any(alert['severity'] == 'critical' for alert in alerts):
            status = 'critical'
        elif any(alert['severity'] == 'warning' for alert in alerts):
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'alerts': alerts,
            'summary': recent_summary,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def export_metrics(self, filepath: str, hours: int = 24) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to export file
            hours: Hours of data to export
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            export_metrics = [
                {
                    'timestamp': m.timestamp,
                    'endpoint': m.endpoint,
                    'method': m.method,
                    'status_code': m.status_code,
                    'response_time_ms': m.response_time_ms,
                    'request_size_bytes': m.request_size_bytes,
                    'response_size_bytes': m.response_size_bytes,
                    'user_agent': m.user_agent,
                    'ip_address': m.ip_address,
                    'error_message': m.error_message
                }
                for m in self.metrics
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_metrics, f, indent=2)
        
        logger.info(f"Exported {len(export_metrics)} metrics to {filepath}")


# Global metrics collector instance
_metrics_collector = APIMetricsCollector()


def get_metrics_collector() -> APIMetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


def monitor_api_performance(func):
    """
    Decorator to monitor API endpoint performance.
    
    Args:
        func: FastAPI endpoint function to monitor
        
    Returns:
        Decorated function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        request = None
        status_code = 200
        error_message = None
        
        # Extract request object from args/kwargs
        for arg in args:
            if hasattr(arg, 'method') and hasattr(arg, 'url'):
                request = arg
                break
        
        try:
            # Execute the original function
            response = await func(*args, **kwargs)
            
            # Extract status code from response if available
            if hasattr(response, 'status_code'):
                status_code = response.status_code
            
            return response
            
        except Exception as e:
            status_code = 500
            error_message = str(e)
            logger.error(f"API endpoint error: {error_message}")
            raise
            
        finally:
            # Record metrics
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            if request:
                # Estimate request/response sizes (simplified)
                request_size = len(str(request.url)) + sum(len(f"{k}: {v}") for k, v in request.headers.items())
                response_size = 0  # Would need actual response object to calculate
                
                metrics = APIMetrics(
                    timestamp=datetime.now().isoformat(),
                    endpoint=str(request.url.path),
                    method=request.method,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    request_size_bytes=request_size,
                    response_size_bytes=response_size,
                    user_agent=request.headers.get('user-agent'),
                    ip_address=request.client.host if hasattr(request, 'client') else None,
                    error_message=error_message
                )
                
                _metrics_collector.record_request(metrics)
    
    return wrapper


class SystemResourceMonitor:
    """Monitors system resource usage."""
    
    @staticmethod
    def get_current_usage() -> Dict[str, Any]:
        """
        Get current system resource usage.
        
        Returns:
            System resource usage dictionary
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O (if available)
            try:
                network = psutil.net_io_counters()
                network_stats = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            except:
                network_stats = {'error': 'network_stats_unavailable'}
            
            # Process-specific info
            current_process = psutil.Process()
            process_info = {
                'pid': current_process.pid,
                'memory_mb': current_process.memory_info().rss / 1024 / 1024,
                'cpu_percent': current_process.cpu_percent(),
                'num_threads': current_process.num_threads(),
                'create_time': datetime.fromtimestamp(current_process.create_time()).isoformat()
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                },
                'memory': {
                    'total_gb': round(memory.total / 1024**3, 2),
                    'available_gb': round(memory.available / 1024**3, 2),
                    'used_gb': round(memory.used / 1024**3, 2),
                    'usage_percent': memory.percent
                },
                'disk': {
                    'total_gb': round(disk.total / 1024**3, 2),
                    'free_gb': round(disk.free / 1024**3, 2),
                    'used_gb': round(disk.used / 1024**3, 2),
                    'usage_percent': round((disk.used / disk.total) * 100, 2)
                },
                'network': network_stats,
                'process': process_info
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    @staticmethod
    def check_resource_alerts(usage: Dict[str, Any], 
                            thresholds: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Check for resource usage alerts.
        
        Args:
            usage: System usage dictionary
            thresholds: Alert thresholds
            
        Returns:
            List of alerts
        """
        if 'error' in usage:
            return []
        
        # Default thresholds
        if thresholds is None:
            thresholds = {
                'cpu_percent': 80,
                'memory_percent': 85,
                'disk_percent': 90
            }
        
        alerts = []
        
        # CPU alert
        if usage['cpu']['usage_percent'] > thresholds['cpu_percent']:
            alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'warning',
                'message': f"CPU usage is {usage['cpu']['usage_percent']:.1f}% (threshold: {thresholds['cpu_percent']}%)",
                'current_value': usage['cpu']['usage_percent'],
                'threshold': thresholds['cpu_percent']
            })
        
        # Memory alert
        if usage['memory']['usage_percent'] > thresholds['memory_percent']:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f"Memory usage is {usage['memory']['usage_percent']:.1f}% (threshold: {thresholds['memory_percent']}%)",
                'current_value': usage['memory']['usage_percent'],
                'threshold': thresholds['memory_percent']
            })
        
        # Disk alert
        if usage['disk']['usage_percent'] > thresholds['disk_percent']:
            severity = 'critical' if usage['disk']['usage_percent'] > 95 else 'warning'
            alerts.append({
                'type': 'high_disk_usage',
                'severity': severity,
                'message': f"Disk usage is {usage['disk']['usage_percent']:.1f}% (threshold: {thresholds['disk_percent']}%)",
                'current_value': usage['disk']['usage_percent'],
                'threshold': thresholds['disk_percent']
            })
        
        return alerts


def create_monitoring_dashboard() -> Dict[str, Any]:
    """
    Create comprehensive monitoring dashboard data.
    
    Returns:
        Dashboard data dictionary
    """
    # Get API metrics
    api_metrics = _metrics_collector.get_metrics_summary(minutes=15)
    api_health = _metrics_collector.get_health_status()
    
    # Get system metrics
    system_usage = SystemResourceMonitor.get_current_usage()
    system_alerts = SystemResourceMonitor.check_resource_alerts(system_usage)
    
    # Combine all data
    dashboard_data = {
        'generated_at': datetime.now().isoformat(),
        'overall_status': _determine_overall_status(api_health, system_alerts),
        'api_monitoring': {
            'health_status': api_health,
            'metrics_summary': api_metrics
        },
        'system_monitoring': {
            'resource_usage': system_usage,
            'alerts': system_alerts
        },
        'summary': {
            'total_api_requests': _metrics_collector.total_requests,
            'api_error_count': _metrics_collector.error_count,
            'system_alerts_count': len(system_alerts),
            'uptime_hours': (datetime.now() - _metrics_collector.start_time).total_seconds() / 3600
        }
    }
    
    return dashboard_data


def _determine_overall_status(api_health: Dict[str, Any], 
                            system_alerts: List[Dict[str, Any]]) -> str:
    """Determine overall system status."""
    api_status = api_health.get('status', 'unknown')
    
    # Check for critical system alerts
    has_critical_system_alert = any(alert['severity'] == 'critical' for alert in system_alerts)
    has_warning_system_alert = any(alert['severity'] == 'warning' for alert in system_alerts)
    
    if api_status == 'critical' or has_critical_system_alert:
        return 'critical'
    elif api_status == 'warning' or has_warning_system_alert:
        return 'warning'
    elif api_status == 'healthy':
        return 'healthy'
    else:
        return 'unknown'
