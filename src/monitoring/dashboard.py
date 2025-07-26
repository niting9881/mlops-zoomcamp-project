"""
Monitoring dashboard and reporting system.
"""
import logging
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
from .api_monitoring import get_metrics_collector, SystemResourceMonitor, create_monitoring_dashboard
from .data_quality import DataQualityValidator, DataProfileResult
from .drift_detection import DataDriftDetector, ModelPerformanceMonitor
from .quality_gates import QualityGateEngine
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Comprehensive monitoring dashboard for MLOps pipeline."""
    
    def __init__(self, 
                 db_path: str = "monitoring.db",
                 quality_gates_db: str = "quality_gates.db",
                 data_quality_db: str = "data_quality.db"):
        """
        Initialize monitoring dashboard.
        
        Args:
            db_path: Path to main monitoring database
            quality_gates_db: Path to quality gates database
            data_quality_db: Path to data quality database
        """
        self.db_path = db_path
        self.quality_gates_db = quality_gates_db
        self.data_quality_db = data_quality_db
        
        # Initialize monitoring components
        self.api_metrics = get_metrics_collector()
        self.system_monitor = SystemResourceMonitor()
        self.quality_validator = DataQualityValidator(data_quality_db)
        self.drift_detector = DataDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()
        self.quality_gates = QualityGateEngine(db_path=quality_gates_db)
        self.metrics_tracker = MetricsTracker(db_path)
        
        self._init_dashboard_database()
        logger.info("Monitoring dashboard initialized")
    
    def _init_dashboard_database(self) -> None:
        """Initialize dashboard-specific database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dashboard_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    report_data TEXT NOT NULL,
                    generated_by TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dashboard_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_name TEXT UNIQUE NOT NULL,
                    config_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
    
    def generate_overview_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive overview dashboard.
        
        Args:
            hours: Time window for data aggregation
            
        Returns:
            Dashboard data dictionary
        """
        logger.info(f"Generating overview dashboard for last {hours} hours")
        
        # Get overall monitoring data
        overall_monitoring = create_monitoring_dashboard()
        
        # Get API metrics summary
        api_summary = self.api_metrics.get_metrics_summary(minutes=hours * 60)
        api_health = self.api_metrics.get_health_status()
        
        # Get system resource usage
        system_usage = self.system_monitor.get_current_usage()
        system_alerts = self.system_monitor.check_resource_alerts(system_usage)
        
        # Get quality gates history
        quality_history = self.quality_gates.get_pipeline_history(days=hours / 24)
        
        # Get data quality reports
        quality_report = self.quality_validator.get_quality_report(hours=hours)
        
        # Get model performance trends
        performance_trends = self._get_performance_trends(hours)
        
        # Calculate overall health score
        health_score = self._calculate_overall_health_score(
            api_health, system_alerts, quality_history, quality_report
        )
        
        dashboard_data = {
            'generated_at': datetime.now().isoformat(),
            'time_window_hours': hours,
            'overall_health': {
                'score': health_score['score'],
                'status': health_score['status'],
                'alerts': health_score['alerts']
            },
            'api_monitoring': {
                'summary': api_summary,
                'health': api_health
            },
            'system_monitoring': {
                'current_usage': system_usage,
                'alerts': system_alerts
            },
            'quality_gates': {
                'history_summary': quality_history['summary'],
                'recent_runs': quality_history['recent_runs'][:5]  # Last 5 runs
            },
            'data_quality': {
                'report_summary': quality_report['summary'],
                'recent_failures': quality_report['recent_failures'][:5]
            },
            'model_performance': performance_trends,
            'key_metrics': self._extract_key_metrics(
                api_summary, system_usage, quality_history, quality_report
            )
        }
        
        # Store dashboard report
        self._store_dashboard_report('overview', dashboard_data)
        
        return dashboard_data
    
    def generate_api_monitoring_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """Generate detailed API monitoring dashboard."""
        logger.info(f"Generating API monitoring dashboard for last {hours} hours")
        
        # Get detailed API metrics
        api_summary = self.api_metrics.get_metrics_summary(minutes=hours * 60)
        api_health = self.api_metrics.get_health_status()
        
        # Get hourly breakdown
        hourly_metrics = self._get_hourly_api_metrics(hours)
        
        # Generate charts
        charts = self._generate_api_charts(api_summary, hourly_metrics)
        
        dashboard_data = {
            'generated_at': datetime.now().isoformat(),
            'time_window_hours': hours,
            'summary': api_summary,
            'health_status': api_health,
            'hourly_breakdown': hourly_metrics,
            'charts': charts,
            'alerts_and_recommendations': self._generate_api_recommendations(api_summary, api_health)
        }
        
        self._store_dashboard_report('api_monitoring', dashboard_data)
        return dashboard_data
    
    def generate_data_quality_dashboard(self, dataset_name: str = None, 
                                      days: int = 7) -> Dict[str, Any]:
        """Generate detailed data quality dashboard."""
        logger.info(f"Generating data quality dashboard for {days} days")
        
        # Get quality report
        quality_report = self.quality_validator.get_quality_report(dataset_name, hours=days * 24)
        
        # Get quality trends
        quality_trends = self._get_quality_trends(dataset_name, days)
        
        # Generate quality charts
        charts = self._generate_quality_charts(quality_report, quality_trends)
        
        dashboard_data = {
            'generated_at': datetime.now().isoformat(),
            'time_window_days': days,
            'dataset_filter': dataset_name,
            'quality_report': quality_report,
            'quality_trends': quality_trends,
            'charts': charts,
            'recommendations': self._generate_quality_recommendations(quality_report)
        }
        
        self._store_dashboard_report('data_quality', dashboard_data)
        return dashboard_data
    
    def generate_model_performance_dashboard(self, model_name: str = None,
                                           days: int = 7) -> Dict[str, Any]:
        """Generate detailed model performance dashboard."""
        logger.info(f"Generating model performance dashboard for {days} days")
        
        # Get performance metrics
        performance_data = self._get_model_performance_data(model_name, days)
        
        # Get drift analysis
        drift_analysis = self._get_drift_analysis_data(days)
        
        # Generate performance charts
        charts = self._generate_performance_charts(performance_data, drift_analysis)
        
        dashboard_data = {
            'generated_at': datetime.now().isoformat(),
            'time_window_days': days,
            'model_filter': model_name,
            'performance_metrics': performance_data,
            'drift_analysis': drift_analysis,
            'charts': charts,
            'alerts': self._generate_performance_alerts(performance_data, drift_analysis)
        }
        
        self._store_dashboard_report('model_performance', dashboard_data)
        return dashboard_data
    
    def generate_quality_gates_dashboard(self, pipeline_name: str = None,
                                       days: int = 7) -> Dict[str, Any]:
        """Generate quality gates monitoring dashboard."""
        logger.info(f"Generating quality gates dashboard for {days} days")
        
        # Get pipeline history
        pipeline_history = self.quality_gates.get_pipeline_history(pipeline_name, days)
        
        # Get gate performance analysis
        gate_analysis = self._get_gate_performance_analysis(pipeline_name, days)
        
        # Generate quality gates charts
        charts = self._generate_quality_gates_charts(pipeline_history, gate_analysis)
        
        dashboard_data = {
            'generated_at': datetime.now().isoformat(),
            'time_window_days': days,
            'pipeline_filter': pipeline_name,
            'pipeline_history': pipeline_history,
            'gate_analysis': gate_analysis,
            'charts': charts,
            'recommendations': self._generate_gates_recommendations(pipeline_history, gate_analysis)
        }
        
        self._store_dashboard_report('quality_gates', dashboard_data)
        return dashboard_data
    
    def _calculate_overall_health_score(self, api_health: Dict[str, Any],
                                      system_alerts: List[Dict[str, Any]],
                                      quality_history: Dict[str, Any],
                                      quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score."""
        scores = []
        alerts = []
        
        # API health score (40% weight)
        api_status = api_health.get('status', 'unknown')
        if api_status == 'healthy':
            api_score = 100
        elif api_status == 'warning':
            api_score = 70
        elif api_status == 'critical':
            api_score = 30
        else:
            api_score = 50
        
        scores.append(('api', api_score, 0.4))
        
        # System health score (20% weight)
        critical_system_alerts = [a for a in system_alerts if a['severity'] == 'critical']
        warning_system_alerts = [a for a in system_alerts if a['severity'] == 'warning']
        
        if len(critical_system_alerts) > 0:
            system_score = 30
            alerts.extend(critical_system_alerts)
        elif len(warning_system_alerts) > 0:
            system_score = 70
            alerts.extend(warning_system_alerts)
        else:
            system_score = 100
        
        scores.append(('system', system_score, 0.2))
        
        # Quality gates score (25% weight)
        gates_summary = quality_history.get('summary', {})
        gates_success_rate = gates_summary.get('success_rate_percent', 0)
        scores.append(('quality_gates', gates_success_rate, 0.25))
        
        # Data quality score (15% weight)
        quality_summary = quality_report.get('summary', {})
        quality_pass_rate = quality_summary.get('pass_rate_percent', 0)
        scores.append(('data_quality', quality_pass_rate, 0.15))
        
        # Calculate weighted average
        weighted_score = sum(score * weight for _, score, weight in scores)
        
        # Determine status
        if weighted_score >= 90:
            status = 'excellent'
        elif weighted_score >= 80:
            status = 'good'
        elif weighted_score >= 70:
            status = 'warning'
        elif weighted_score >= 50:
            status = 'poor'
        else:
            status = 'critical'
        
        return {
            'score': round(weighted_score, 1),
            'status': status,
            'component_scores': {name: score for name, score, _ in scores},
            'alerts': alerts[:5]  # Top 5 alerts
        }
    
    def _get_hourly_api_metrics(self, hours: int) -> List[Dict[str, Any]]:
        """Get hourly breakdown of API metrics."""
        # This would query the API metrics database for hourly aggregations
        # For now, return mock data structure
        hourly_data = []
        
        for hour in range(hours):
            timestamp = datetime.now() - timedelta(hours=hour)
            hourly_data.append({
                'hour': timestamp.strftime('%Y-%m-%d %H:00'),
                'request_count': 0,  # Would be calculated from actual data
                'error_rate': 0.0,
                'avg_response_time': 0.0,
                'p95_response_time': 0.0
            })
        
        return hourly_data
    
    def _get_performance_trends(self, hours: int) -> Dict[str, Any]:
        """Get model performance trends."""
        # Query metrics tracker for performance data
        try:
            # Get performance metrics from database
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.metrics_tracker.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, metric_name, metric_value 
                    FROM metrics 
                    WHERE timestamp >= ? AND metric_name IN ('accuracy', 'precision', 'recall', 'f1_score')
                    ORDER BY timestamp DESC
                """, (cutoff_time.isoformat(),))
                
                results = cursor.fetchall()
            
            # Process results
            metrics_by_time = {}
            for timestamp, metric_name, metric_value in results:
                if timestamp not in metrics_by_time:
                    metrics_by_time[timestamp] = {}
                metrics_by_time[timestamp][metric_name] = metric_value
            
            # Calculate trends
            if len(metrics_by_time) > 1:
                timestamps = sorted(metrics_by_time.keys())
                latest_metrics = metrics_by_time[timestamps[-1]]
                previous_metrics = metrics_by_time[timestamps[-2]] if len(timestamps) > 1 else {}
                
                trends = {}
                for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                    current = latest_metrics.get(metric_name, 0)
                    previous = previous_metrics.get(metric_name, 0)
                    change = current - previous if previous > 0 else 0
                    trends[metric_name] = {
                        'current': current,
                        'previous': previous,
                        'change': change,
                        'trend': 'up' if change > 0 else 'down' if change < 0 else 'stable'
                    }
            else:
                trends = {}
            
            return {
                'has_data': len(metrics_by_time) > 0,
                'total_data_points': len(metrics_by_time),
                'trends': trends,
                'latest_timestamp': timestamps[-1] if timestamps else None
            }
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {str(e)}")
            return {'has_data': False, 'error': str(e)}
    
    def _extract_key_metrics(self, api_summary: Dict, system_usage: Dict,
                           quality_history: Dict, quality_report: Dict) -> Dict[str, Any]:
        """Extract key metrics for overview display."""
        key_metrics = {}
        
        # API metrics
        if not api_summary.get('no_data', False):
            key_metrics['api_requests_per_minute'] = api_summary.get('requests_per_minute', 0)
            key_metrics['api_error_rate'] = api_summary.get('error_rate_percent', 0)
            key_metrics['api_avg_response_time'] = api_summary.get('response_time_stats', {}).get('mean_ms', 0)
        
        # System metrics
        if 'error' not in system_usage:
            key_metrics['cpu_usage'] = system_usage.get('cpu', {}).get('usage_percent', 0)
            key_metrics['memory_usage'] = system_usage.get('memory', {}).get('usage_percent', 0)
            key_metrics['disk_usage'] = system_usage.get('disk', {}).get('usage_percent', 0)
        
        # Quality metrics
        quality_summary = quality_history.get('summary', {})
        key_metrics['quality_gates_success_rate'] = quality_summary.get('success_rate_percent', 0)
        
        data_quality_summary = quality_report.get('summary', {})
        key_metrics['data_quality_pass_rate'] = data_quality_summary.get('pass_rate_percent', 0)
        
        return key_metrics
    
    def _generate_api_charts(self, api_summary: Dict, hourly_metrics: List) -> Dict[str, str]:
        """Generate API monitoring charts."""
        charts = {}
        
        try:
            # Response time distribution chart
            if not api_summary.get('no_data', False):
                response_stats = api_summary.get('response_time_stats', {})
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Min', 'Mean', 'P50', 'P95', 'P99', 'Max'],
                    y=[
                        response_stats.get('min_ms', 0),
                        response_stats.get('mean_ms', 0),
                        response_stats.get('p50_ms', 0),
                        response_stats.get('p95_ms', 0),
                        response_stats.get('p99_ms', 0),
                        response_stats.get('max_ms', 0)
                    ],
                    name='Response Time (ms)'
                ))
                
                fig.update_layout(
                    title='API Response Time Distribution',
                    xaxis_title='Percentile',
                    yaxis_title='Response Time (ms)'
                )
                
                charts['response_time_distribution'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Hourly request volume (if data available)
            if hourly_metrics:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[h['hour'] for h in hourly_metrics],
                    y=[h['request_count'] for h in hourly_metrics],
                    mode='lines+markers',
                    name='Request Count'
                ))
                
                fig.update_layout(
                    title='Hourly Request Volume',
                    xaxis_title='Time',
                    yaxis_title='Request Count'
                )
                
                charts['hourly_requests'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error generating API charts: {str(e)}")
            charts['error'] = f"Chart generation failed: {str(e)}"
        
        return charts
    
    def _generate_quality_charts(self, quality_report: Dict, quality_trends: Dict) -> Dict[str, str]:
        """Generate data quality charts."""
        charts = {}
        
        try:
            # Quality checks breakdown by severity
            breakdown = quality_report.get('breakdown', {})
            severity_data = breakdown.get('by_severity', {})
            
            if severity_data:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(severity_data.keys()),
                        values=list(severity_data.values()),
                        hole=0.3
                    )
                ])
                
                fig.update_layout(title='Quality Checks by Severity')
                charts['severity_breakdown'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Quality pass rate over time (if trends available)
            if quality_trends and 'pass_rate_over_time' in quality_trends:
                trend_data = quality_trends['pass_rate_over_time']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[d['timestamp'] for d in trend_data],
                    y=[d['pass_rate'] for d in trend_data],
                    mode='lines+markers',
                    name='Pass Rate %'
                ))
                
                fig.update_layout(
                    title='Data Quality Pass Rate Over Time',
                    xaxis_title='Time',
                    yaxis_title='Pass Rate (%)'
                )
                
                charts['pass_rate_trend'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        except Exception as e:
            logger.error(f"Error generating quality charts: {str(e)}")
            charts['error'] = f"Chart generation failed: {str(e)}"
        
        return charts
    
    def _generate_performance_charts(self, performance_data: Dict, drift_analysis: Dict) -> Dict[str, str]:
        """Generate model performance charts."""
        charts = {}
        
        try:
            # Performance metrics over time
            if performance_data.get('has_data', False):
                trends = performance_data.get('trends', {})
                
                if trends:
                    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                    fig = go.Figure()
                    
                    for metric in metrics:
                        if metric in trends:
                            fig.add_trace(go.Bar(
                                name=metric.title(),
                                x=[metric],
                                y=[trends[metric]['current']]
                            ))
                    
                    fig.update_layout(
                        title='Current Model Performance Metrics',
                        xaxis_title='Metric',
                        yaxis_title='Score'
                    )
                    
                    charts['performance_metrics'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        except Exception as e:
            logger.error(f"Error generating performance charts: {str(e)}")
            charts['error'] = f"Chart generation failed: {str(e)}"
        
        return charts
    
    def _generate_quality_gates_charts(self, pipeline_history: Dict, gate_analysis: Dict) -> Dict[str, str]:
        """Generate quality gates charts."""
        charts = {}
        
        try:
            # Pipeline success rate over time
            recent_runs = pipeline_history.get('recent_runs', [])
            
            if recent_runs:
                fig = go.Figure()
                
                success_data = []
                timestamps = []
                
                for run in reversed(recent_runs[-10:]):  # Last 10 runs
                    timestamps.append(run['timestamp'])
                    success_rate = (run['passed_gates'] / run['total_gates']) * 100 if run['total_gates'] > 0 else 0
                    success_data.append(success_rate)
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=success_data,
                    mode='lines+markers',
                    name='Success Rate %'
                ))
                
                fig.update_layout(
                    title='Quality Gates Success Rate Trend',
                    xaxis_title='Pipeline Run',
                    yaxis_title='Success Rate (%)'
                )
                
                charts['success_rate_trend'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        except Exception as e:
            logger.error(f"Error generating quality gates charts: {str(e)}")
            charts['error'] = f"Chart generation failed: {str(e)}"
        
        return charts
    
    def _generate_api_recommendations(self, api_summary: Dict, api_health: Dict) -> List[Dict[str, str]]:
        """Generate API monitoring recommendations."""
        recommendations = []
        
        if api_summary.get('no_data', False):
            recommendations.append({
                'type': 'info',
                'title': 'No API Data',
                'message': 'No API metrics available. Ensure monitoring is properly configured.'
            })
            return recommendations
        
        # Check error rate
        error_rate = api_summary.get('error_rate_percent', 0)
        if error_rate > 10:
            recommendations.append({
                'type': 'critical',
                'title': 'High Error Rate',
                'message': f'API error rate is {error_rate:.1f}%. Investigate error logs and fix issues.'
            })
        elif error_rate > 5:
            recommendations.append({
                'type': 'warning',
                'title': 'Elevated Error Rate',
                'message': f'API error rate is {error_rate:.1f}%. Monitor closely and consider optimization.'
            })
        
        # Check response times
        response_stats = api_summary.get('response_time_stats', {})
        p95_time = response_stats.get('p95_ms', 0)
        
        if p95_time > 5000:
            recommendations.append({
                'type': 'critical',
                'title': 'Slow Response Times',
                'message': f'95th percentile response time is {p95_time:.0f}ms. Optimize performance.'
            })
        elif p95_time > 2000:
            recommendations.append({
                'type': 'warning',
                'title': 'Response Time Warning',
                'message': f'95th percentile response time is {p95_time:.0f}ms. Consider optimization.'
            })
        
        return recommendations
    
    def _generate_quality_recommendations(self, quality_report: Dict) -> List[Dict[str, str]]:
        """Generate data quality recommendations."""
        recommendations = []
        
        summary = quality_report.get('summary', {})
        pass_rate = summary.get('pass_rate_percent', 0)
        
        if pass_rate < 80:
            recommendations.append({
                'type': 'critical',
                'title': 'Low Data Quality',
                'message': f'Data quality pass rate is {pass_rate:.1f}%. Review and fix data issues.'
            })
        elif pass_rate < 95:
            recommendations.append({
                'type': 'warning',
                'title': 'Data Quality Issues',
                'message': f'Data quality pass rate is {pass_rate:.1f}%. Monitor and improve data quality.'
            })
        
        # Check recent failures
        recent_failures = quality_report.get('recent_failures', [])
        if recent_failures:
            critical_failures = [f for f in recent_failures if f['severity'] == 'critical']
            if critical_failures:
                recommendations.append({
                    'type': 'critical',
                    'title': 'Critical Data Quality Failures',
                    'message': f'Found {len(critical_failures)} critical data quality failures. Address immediately.'
                })
        
        return recommendations
    
    def _generate_performance_alerts(self, performance_data: Dict, drift_analysis: Dict) -> List[Dict[str, str]]:
        """Generate model performance alerts."""
        alerts = []
        
        if not performance_data.get('has_data', False):
            alerts.append({
                'type': 'info',
                'title': 'No Performance Data',
                'message': 'No recent model performance data available.'
            })
            return alerts
        
        trends = performance_data.get('trends', {})
        
        # Check for declining performance
        for metric_name, trend_data in trends.items():
            if trend_data['trend'] == 'down' and abs(trend_data['change']) > 0.05:  # 5% decline
                alerts.append({
                    'type': 'warning',
                    'title': f'Declining {metric_name.title()}',
                    'message': f'{metric_name.title()} decreased by {abs(trend_data["change"]):.3f}'
                })
        
        return alerts
    
    def _generate_gates_recommendations(self, pipeline_history: Dict, gate_analysis: Dict) -> List[Dict[str, str]]:
        """Generate quality gates recommendations."""
        recommendations = []
        
        summary = pipeline_history.get('summary', {})
        success_rate = summary.get('success_rate_percent', 0)
        
        if success_rate < 80:
            recommendations.append({
                'type': 'critical',
                'title': 'Low Pipeline Success Rate',
                'message': f'Pipeline success rate is {success_rate:.1f}%. Review failing gates.'
            })
        elif success_rate < 95:
            recommendations.append({
                'type': 'warning',
                'title': 'Pipeline Success Issues',
                'message': f'Pipeline success rate is {success_rate:.1f}%. Monitor gate configurations.'
            })
        
        return recommendations
    
    def _get_quality_trends(self, dataset_name: str, days: int) -> Dict[str, Any]:
        """Get data quality trends over time."""
        # Mock implementation - would query actual database
        return {'pass_rate_over_time': []}
    
    def _get_model_performance_data(self, model_name: str, days: int) -> Dict[str, Any]:
        """Get model performance data."""
        return self._get_performance_trends(days * 24)
    
    def _get_drift_analysis_data(self, days: int) -> Dict[str, Any]:
        """Get drift analysis data."""
        # Mock implementation
        return {'has_drift_data': False}
    
    def _get_gate_performance_analysis(self, pipeline_name: str, days: int) -> Dict[str, Any]:
        """Get gate performance analysis."""
        # Mock implementation
        return {'gate_success_rates': {}}
    
    def _store_dashboard_report(self, report_type: str, report_data: Dict[str, Any]) -> None:
        """Store dashboard report in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO dashboard_reports (timestamp, report_type, report_data, generated_by)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                report_type,
                json.dumps(report_data),
                'MonitoringDashboard'
            ))
    
    def export_dashboard_report(self, report_type: str, filepath: str, 
                              format: str = 'json') -> None:
        """
        Export dashboard report to file.
        
        Args:
            report_type: Type of report to export
            filepath: Output file path
            format: Export format ('json', 'html')
        """
        # Get latest report of specified type
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT report_data FROM dashboard_reports 
                WHERE report_type = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (report_type,))
            
            result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"No reports found for type: {report_type}")
        
        report_data = json.loads(result[0])
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
        elif format.lower() == 'html':
            html_content = self._generate_html_report(report_type, report_data)
            with open(output_path, 'w') as f:
                f.write(html_content)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dashboard report exported to {filepath}")
    
    def _generate_html_report(self, report_type: str, report_data: Dict[str, Any]) -> str:
        """Generate HTML report from dashboard data."""
        # Simple HTML template - can be enhanced with proper templating
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLOps Monitoring Report - {report_type.title()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .alert {{ padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .alert.critical {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
                .alert.warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
                .alert.info {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MLOps Monitoring Report</h1>
                <h2>{report_type.replace('_', ' ').title()}</h2>
                <p>Generated at: {report_data.get('generated_at', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h3>Summary</h3>
                <pre>{json.dumps(report_data, indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        return html
