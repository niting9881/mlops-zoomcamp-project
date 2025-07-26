"""
Automated quality gates and validation pipeline.
"""
import logging
import yaml
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import sqlite3
import numpy as np
from .data_quality import DataQualityValidator, DataQualityResult
from .drift_detection import DataDriftDetector, ModelPerformanceMonitor
from .metrics import MetricsTracker, PerformanceAlertManager

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Quality gate status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class GateSeverity(Enum):
    """Quality gate severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_name: str
    gate_type: str
    status: GateStatus
    severity: GateSeverity
    score: float
    threshold: float
    message: str
    details: Dict[str, Any]
    execution_time_ms: float
    timestamp: str


@dataclass
class QualityGateConfig:
    """Configuration for a quality gate."""
    name: str
    type: str
    enabled: bool
    severity: GateSeverity
    threshold: float
    parameters: Dict[str, Any]
    depends_on: List[str]
    description: str


class QualityGateEngine:
    """Engine for executing quality gates in automated pipelines."""
    
    def __init__(self, config_path: str = None, db_path: str = "quality_gates.db"):
        """
        Initialize quality gate engine.
        
        Args:
            config_path: Path to quality gates configuration file
            db_path: Path to SQLite database for storing results
        """
        self.config_path = config_path
        self.db_path = db_path
        self.gates: Dict[str, QualityGateConfig] = {}
        self.data_quality_validator = DataQualityValidator()
        self.drift_detector = DataDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()
        self.metrics_tracker = MetricsTracker()
        self.alert_manager = PerformanceAlertManager()
        
        self._init_database()
        if config_path:
            self.load_configuration(config_path)
        
        logger.info(f"Quality gate engine initialized with database: {db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for storing gate results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gate_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pipeline_run_id TEXT,
                    gate_name TEXT NOT NULL,
                    gate_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT NOT NULL,
                    execution_time_ms REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    pipeline_name TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    total_gates INTEGER NOT NULL,
                    passed_gates INTEGER NOT NULL,
                    failed_gates INTEGER NOT NULL,
                    warning_gates INTEGER NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    metadata TEXT
                )
            """)
    
    def load_configuration(self, config_path: str) -> None:
        """
        Load quality gates configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self.gates = {}
            for gate_config in config_data.get('quality_gates', []):
                gate = QualityGateConfig(
                    name=gate_config['name'],
                    type=gate_config['type'],
                    enabled=gate_config.get('enabled', True),
                    severity=GateSeverity(gate_config.get('severity', 'medium')),
                    threshold=gate_config.get('threshold', 0.0),
                    parameters=gate_config.get('parameters', {}),
                    depends_on=gate_config.get('depends_on', []),
                    description=gate_config.get('description', '')
                )
                self.gates[gate.name] = gate
            
            logger.info(f"Loaded {len(self.gates)} quality gates from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def create_default_configuration(self, output_path: str) -> None:
        """
        Create default quality gates configuration file.
        
        Args:
            output_path: Path to save configuration file
        """
        default_config = {
            'quality_gates': [
                {
                    'name': 'data_completeness',
                    'type': 'data_quality',
                    'enabled': True,
                    'severity': 'critical',
                    'threshold': 95.0,
                    'parameters': {
                        'max_missing_percent': 5.0,
                        'required_columns': []
                    },
                    'description': 'Validates data completeness and missing values'
                },
                {
                    'name': 'data_consistency',
                    'type': 'data_quality',
                    'enabled': True,
                    'severity': 'high',
                    'threshold': 98.0,
                    'parameters': {
                        'max_duplicate_percent': 2.0,
                        'consistency_rules': {}
                    },
                    'description': 'Validates data consistency and duplicates'
                },
                {
                    'name': 'data_drift_detection',
                    'type': 'drift_detection',
                    'enabled': True,
                    'severity': 'medium',
                    'threshold': 0.05,
                    'parameters': {
                        'statistical_threshold': 0.05,
                        'drift_methods': ['ks_test', 'chi2_test']
                    },
                    'description': 'Detects statistical drift in input data'
                },
                {
                    'name': 'model_performance',
                    'type': 'model_performance',
                    'enabled': True,
                    'severity': 'critical',
                    'threshold': 0.85,
                    'parameters': {
                        'metrics': ['accuracy', 'precision', 'recall'],
                        'baseline_comparison': True
                    },
                    'description': 'Validates model performance against thresholds'
                },
                {
                    'name': 'prediction_confidence',
                    'type': 'prediction_quality',
                    'enabled': True,
                    'severity': 'medium',
                    'threshold': 0.7,
                    'parameters': {
                        'min_confidence': 0.7,
                        'confidence_distribution_check': True
                    },
                    'description': 'Validates prediction confidence scores'
                },
                {
                    'name': 'data_volume',
                    'type': 'data_volume',
                    'enabled': True,
                    'severity': 'high',
                    'threshold': 100,
                    'parameters': {
                        'min_rows': 100,
                        'expected_volume_variance': 0.2
                    },
                    'description': 'Validates minimum data volume requirements'
                },
                {
                    'name': 'feature_importance_stability',
                    'type': 'feature_analysis',
                    'enabled': True,
                    'severity': 'medium',
                    'threshold': 0.8,
                    'parameters': {
                        'importance_correlation_threshold': 0.8,
                        'top_features_consistency': 10
                    },
                    'description': 'Validates stability of feature importance'
                }
            ]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            if output_file.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            else:
                json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default configuration at {output_path}")
    
    def execute_gate(self, gate_name: str, 
                    data: Dict[str, Any],
                    context: Dict[str, Any] = None) -> QualityGateResult:
        """
        Execute a specific quality gate.
        
        Args:
            gate_name: Name of the gate to execute
            data: Input data for the gate
            context: Additional context information
            
        Returns:
            Quality gate result
        """
        if gate_name not in self.gates:
            raise ValueError(f"Gate '{gate_name}' not found in configuration")
        
        gate = self.gates[gate_name]
        
        if not gate.enabled:
            return QualityGateResult(
                gate_name=gate_name,
                gate_type=gate.type,
                status=GateStatus.SKIPPED,
                severity=gate.severity,
                score=0.0,
                threshold=gate.threshold,
                message="Gate is disabled",
                details={},
                execution_time_ms=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        start_time = datetime.now()
        
        try:
            # Execute gate based on type
            if gate.type == 'data_quality':
                result = self._execute_data_quality_gate(gate, data, context)
            elif gate.type == 'drift_detection':
                result = self._execute_drift_detection_gate(gate, data, context)
            elif gate.type == 'model_performance':
                result = self._execute_model_performance_gate(gate, data, context)
            elif gate.type == 'prediction_quality':
                result = self._execute_prediction_quality_gate(gate, data, context)
            elif gate.type == 'data_volume':
                result = self._execute_data_volume_gate(gate, data, context)
            elif gate.type == 'feature_analysis':
                result = self._execute_feature_analysis_gate(gate, data, context)
            else:
                raise ValueError(f"Unknown gate type: {gate.type}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error executing gate '{gate_name}': {str(e)}")
            
            return QualityGateResult(
                gate_name=gate_name,
                gate_type=gate.type,
                status=GateStatus.FAILED,
                severity=gate.severity,
                score=0.0,
                threshold=gate.threshold,
                message=f"Gate execution failed: {str(e)}",
                details={'error': str(e)},
                execution_time_ms=execution_time,
                timestamp=datetime.now().isoformat()
            )
    
    def _execute_data_quality_gate(self, gate: QualityGateConfig, 
                                 data: Dict[str, Any], 
                                 context: Dict[str, Any] = None) -> QualityGateResult:
        """Execute data quality gate."""
        df = data.get('dataframe')
        if df is None:
            raise ValueError("DataFrame is required for data quality gate")
        
        # Run data quality validation
        validation_results = self.data_quality_validator.run_comprehensive_validation(
            df, 
            dataset_name=data.get('dataset_name', 'unknown'),
            validation_config={
                'max_missing_percent': gate.parameters.get('max_missing_percent', 5.0),
                'required_columns': gate.parameters.get('required_columns', []),
                'consistency_rules': gate.parameters.get('consistency_rules', {})
            }
        )
        
        # Calculate score based on pass rate
        pass_rate = validation_results['summary']['pass_rate_percent']
        status = GateStatus.PASSED if pass_rate >= gate.threshold else GateStatus.FAILED
        
        # Check for warnings
        if status == GateStatus.PASSED and validation_results['summary']['warning_failures'] > 0:
            status = GateStatus.WARNING
        
        return QualityGateResult(
            gate_name=gate.name,
            gate_type=gate.type,
            status=status,
            severity=gate.severity,
            score=pass_rate,
            threshold=gate.threshold,
            message=f"Data quality check: {pass_rate:.1f}% pass rate",
            details=validation_results,
            execution_time_ms=0.0,  # Will be set by caller
            timestamp=datetime.now().isoformat()
        )
    
    def _execute_drift_detection_gate(self, gate: QualityGateConfig,
                                    data: Dict[str, Any],
                                    context: Dict[str, Any] = None) -> QualityGateResult:
        """Execute drift detection gate."""
        current_data = data.get('current_data')
        reference_data = data.get('reference_data')
        
        if current_data is None or reference_data is None:
            raise ValueError("Both current_data and reference_data are required for drift detection")
        
        # Fit reference data if detector is not fitted
        if not hasattr(self.drift_detector, 'reference_statistics') or self.drift_detector.reference_statistics is None:
            self.drift_detector.fit_reference(reference_data)
        
        # Detect drift
        drift_results = self.drift_detector.detect_drift(current_data)
        
        # Calculate overall drift score (1 - max p-value, so higher score means more drift)
        p_values = [result.get('p_value', 1.0) for result in drift_results.values()]
        min_p_value = min(p_values) if p_values else 1.0
        drift_score = 1.0 - min_p_value
        
        # Determine status based on threshold (threshold is minimum p-value)
        status = GateStatus.FAILED if min_p_value < gate.threshold else GateStatus.PASSED
        
        # Check for warnings (some columns drifted but not all)
        if status == GateStatus.PASSED:
            warning_threshold = gate.parameters.get('warning_threshold', gate.threshold * 2)
            if any(p < warning_threshold for p in p_values):
                status = GateStatus.WARNING
        
        return QualityGateResult(
            gate_name=gate.name,
            gate_type=gate.type,
            status=status,
            severity=gate.severity,
            score=drift_score,
            threshold=gate.threshold,
            message=f"Drift detection: minimum p-value = {min_p_value:.4f}",
            details=drift_results,
            execution_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def _execute_model_performance_gate(self, gate: QualityGateConfig,
                                      data: Dict[str, Any],
                                      context: Dict[str, Any] = None) -> QualityGateResult:
        """Execute model performance gate."""
        y_true = data.get('y_true')
        y_pred = data.get('y_pred')
        
        if y_true is None or y_pred is None:
            raise ValueError("Both y_true and y_pred are required for model performance gate")
        
        # Evaluate performance
        performance_results = self.performance_monitor.evaluate_batch(y_true, y_pred)
        
        # Get primary metric score
        primary_metric = gate.parameters.get('primary_metric', 'accuracy')
        score = performance_results['metrics'].get(primary_metric, 0.0)
        
        # Determine status
        status = GateStatus.PASSED if score >= gate.threshold else GateStatus.FAILED
        
        # Check for warnings based on other metrics
        if status == GateStatus.PASSED:
            warning_metrics = gate.parameters.get('warning_metrics', {})
            for metric_name, min_value in warning_metrics.items():
                if performance_results['metrics'].get(metric_name, 0.0) < min_value:
                    status = GateStatus.WARNING
                    break
        
        return QualityGateResult(
            gate_name=gate.name,
            gate_type=gate.type,
            status=status,
            severity=gate.severity,
            score=score,
            threshold=gate.threshold,
            message=f"Model performance: {primary_metric} = {score:.3f}",
            details=performance_results,
            execution_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def _execute_prediction_quality_gate(self, gate: QualityGateConfig,
                                       data: Dict[str, Any],
                                       context: Dict[str, Any] = None) -> QualityGateResult:
        """Execute prediction quality gate."""
        predictions = data.get('predictions')
        confidence_scores = data.get('confidence_scores')
        
        if predictions is None:
            raise ValueError("Predictions are required for prediction quality gate")
        
        details = {}
        
        if confidence_scores is not None:
            # Analyze confidence scores
            confidence_array = np.array(confidence_scores)
            min_confidence = gate.parameters.get('min_confidence', 0.5)
            
            # Calculate percentage of predictions above minimum confidence
            high_confidence_ratio = np.mean(confidence_array >= min_confidence)
            
            details.update({
                'confidence_analysis': {
                    'mean_confidence': float(np.mean(confidence_array)),
                    'min_confidence': float(np.min(confidence_array)),
                    'max_confidence': float(np.max(confidence_array)),
                    'std_confidence': float(np.std(confidence_array)),
                    'high_confidence_ratio': high_confidence_ratio,
                    'threshold': min_confidence
                }
            })
            
            score = high_confidence_ratio
        else:
            # If no confidence scores, use prediction diversity as a proxy for quality
            unique_predictions = len(np.unique(predictions))
            total_predictions = len(predictions)
            diversity_ratio = unique_predictions / total_predictions if total_predictions > 0 else 0
            
            details.update({
                'prediction_analysis': {
                    'total_predictions': total_predictions,
                    'unique_predictions': unique_predictions,
                    'diversity_ratio': diversity_ratio
                }
            })
            
            score = diversity_ratio
        
        # Determine status
        status = GateStatus.PASSED if score >= gate.threshold else GateStatus.FAILED
        
        return QualityGateResult(
            gate_name=gate.name,
            gate_type=gate.type,
            status=status,
            severity=gate.severity,
            score=score,
            threshold=gate.threshold,
            message=f"Prediction quality: score = {score:.3f}",
            details=details,
            execution_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def _execute_data_volume_gate(self, gate: QualityGateConfig,
                                data: Dict[str, Any],
                                context: Dict[str, Any] = None) -> QualityGateResult:
        """Execute data volume gate."""
        df = data.get('dataframe')
        if df is None:
            raise ValueError("DataFrame is required for data volume gate")
        
        actual_rows = len(df)
        min_rows = gate.parameters.get('min_rows', gate.threshold)
        
        # Calculate score as ratio of actual to minimum required
        score = actual_rows / min_rows if min_rows > 0 else 1.0
        
        # Determine status
        status = GateStatus.PASSED if actual_rows >= min_rows else GateStatus.FAILED
        
        # Check expected volume variance
        expected_rows = context.get('expected_rows') if context else None
        variance_details = {}
        
        if expected_rows:
            variance_threshold = gate.parameters.get('expected_volume_variance', 0.2)
            actual_variance = abs(actual_rows - expected_rows) / expected_rows
            
            variance_details = {
                'expected_rows': expected_rows,
                'actual_rows': actual_rows,
                'variance_ratio': actual_variance,
                'variance_threshold': variance_threshold,
                'variance_within_limits': actual_variance <= variance_threshold
            }
            
            if status == GateStatus.PASSED and actual_variance > variance_threshold:
                status = GateStatus.WARNING
        
        details = {
            'volume_analysis': {
                'actual_rows': actual_rows,
                'minimum_required': min_rows,
                'volume_ratio': score,
                **variance_details
            }
        }
        
        return QualityGateResult(
            gate_name=gate.name,
            gate_type=gate.type,
            status=status,
            severity=gate.severity,
            score=score,
            threshold=gate.threshold,
            message=f"Data volume: {actual_rows} rows (minimum: {min_rows})",
            details=details,
            execution_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def _execute_feature_analysis_gate(self, gate: QualityGateConfig,
                                     data: Dict[str, Any],
                                     context: Dict[str, Any] = None) -> QualityGateResult:
        """Execute feature analysis gate."""
        current_importance = data.get('feature_importance')
        reference_importance = data.get('reference_feature_importance')
        
        if current_importance is None:
            raise ValueError("Feature importance data is required for feature analysis gate")
        
        details = {'feature_analysis': {}}
        score = 1.0  # Default score
        
        if reference_importance is not None:
            # Calculate correlation between current and reference importance
            from scipy.stats import pearsonr
            
            # Align features (in case they're in different order or missing)
            common_features = set(current_importance.keys()) & set(reference_importance.keys())
            
            if len(common_features) > 0:
                current_values = [current_importance[f] for f in common_features]
                reference_values = [reference_importance[f] for f in common_features]
                
                correlation, p_value = pearsonr(current_values, reference_values)
                score = correlation if correlation > 0 else 0.0
                
                details['feature_analysis'].update({
                    'importance_correlation': correlation,
                    'correlation_p_value': p_value,
                    'common_features_count': len(common_features),
                    'current_features_count': len(current_importance),
                    'reference_features_count': len(reference_importance)
                })
                
                # Check top features consistency
                top_n = gate.parameters.get('top_features_consistency', 10)
                current_top = sorted(current_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
                reference_top = sorted(reference_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
                
                current_top_features = set(f[0] for f in current_top)
                reference_top_features = set(f[0] for f in reference_top)
                
                top_features_overlap = len(current_top_features & reference_top_features)
                top_features_consistency = top_features_overlap / top_n if top_n > 0 else 1.0
                
                details['feature_analysis'].update({
                    'top_features_consistency': top_features_consistency,
                    'top_features_overlap_count': top_features_overlap,
                    'current_top_features': list(current_top_features),
                    'reference_top_features': list(reference_top_features)
                })
        else:
            # Without reference, just analyze current feature importance distribution
            importance_values = list(current_importance.values())
            details['feature_analysis'].update({
                'total_features': len(importance_values),
                'mean_importance': float(np.mean(importance_values)),
                'std_importance': float(np.std(importance_values)),
                'max_importance': float(np.max(importance_values)),
                'min_importance': float(np.min(importance_values))
            })
        
        # Determine status
        status = GateStatus.PASSED if score >= gate.threshold else GateStatus.FAILED
        
        return QualityGateResult(
            gate_name=gate.name,
            gate_type=gate.type,
            status=status,
            severity=gate.severity,
            score=score,
            threshold=gate.threshold,
            message=f"Feature analysis: score = {score:.3f}",
            details=details,
            execution_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def execute_pipeline(self, pipeline_name: str,
                        data: Dict[str, Any],
                        context: Dict[str, Any] = None,
                        gates_to_run: List[str] = None) -> Dict[str, Any]:
        """
        Execute a complete quality gate pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            data: Input data for all gates
            context: Additional context information
            gates_to_run: Specific gates to run (if None, run all enabled gates)
            
        Returns:
            Pipeline execution results
        """
        run_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting quality gate pipeline: {pipeline_name} (run_id: {run_id})")
        
        # Determine gates to execute
        if gates_to_run is None:
            gates_to_execute = [name for name, gate in self.gates.items() if gate.enabled]
        else:
            gates_to_execute = [name for name in gates_to_run if name in self.gates and self.gates[name].enabled]
        
        # Sort gates by dependencies
        gates_to_execute = self._sort_gates_by_dependencies(gates_to_execute)
        
        results = []
        failed_gates = []
        warning_gates = []
        
        # Execute gates
        for gate_name in gates_to_execute:
            try:
                logger.info(f"Executing gate: {gate_name}")
                result = self.execute_gate(gate_name, data, context)
                results.append(result)
                
                if result.status == GateStatus.FAILED:
                    failed_gates.append(gate_name)
                    # Check if this is a critical gate that should stop the pipeline
                    if result.severity == GateSeverity.CRITICAL:
                        logger.warning(f"Critical gate '{gate_name}' failed. Stopping pipeline execution.")
                        break
                elif result.status == GateStatus.WARNING:
                    warning_gates.append(gate_name)
                
            except Exception as e:
                logger.error(f"Error executing gate '{gate_name}': {str(e)}")
                failed_gates.append(gate_name)
                
                # Create error result
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    gate_type=self.gates[gate_name].type,
                    status=GateStatus.FAILED,
                    severity=self.gates[gate_name].severity,
                    score=0.0,
                    threshold=self.gates[gate_name].threshold,
                    message=f"Gate execution error: {str(e)}",
                    details={'error': str(e)},
                    execution_time_ms=0.0,
                    timestamp=datetime.now().isoformat()
                )
                results.append(error_result)
        
        # Calculate overall pipeline status
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.status == GateStatus.PASSED)
        
        if len(failed_gates) == 0:
            if len(warning_gates) == 0:
                overall_status = "passed"
            else:
                overall_status = "passed_with_warnings"
        else:
            overall_status = "failed"
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Store pipeline run results
        self._store_pipeline_run(run_id, pipeline_name, overall_status, results, execution_time, context)
        
        pipeline_results = {
            'run_id': run_id,
            'pipeline_name': pipeline_name,
            'execution_timestamp': start_time.isoformat(),
            'overall_status': overall_status,
            'execution_time_ms': execution_time,
            'summary': {
                'total_gates': total_gates,
                'passed_gates': passed_gates,
                'failed_gates': len(failed_gates),
                'warning_gates': len(warning_gates),
                'skipped_gates': sum(1 for r in results if r.status == GateStatus.SKIPPED)
            },
            'failed_gates': failed_gates,
            'warning_gates': warning_gates,
            'gate_results': [
                {
                    'gate_name': r.gate_name,
                    'gate_type': r.gate_type,
                    'status': r.status.value,
                    'severity': r.severity.value,
                    'score': r.score,
                    'threshold': r.threshold,
                    'message': r.message,
                    'execution_time_ms': r.execution_time_ms,
                    'timestamp': r.timestamp
                }
                for r in results
            ],
            'detailed_results': results
        }
        
        logger.info(f"Pipeline completed: {pipeline_name} - Status: {overall_status} - "
                   f"Gates: {passed_gates}/{total_gates} passed")
        
        # Generate alerts for failed critical gates
        if failed_gates:
            critical_failures = [r for r in results if r.status == GateStatus.FAILED and r.severity == GateSeverity.CRITICAL]
            if critical_failures:
                self._generate_critical_failure_alerts(run_id, critical_failures)
        
        return pipeline_results
    
    def _sort_gates_by_dependencies(self, gate_names: List[str]) -> List[str]:
        """Sort gates by their dependencies using topological sort."""
        # Simple dependency resolution - can be enhanced for complex scenarios
        sorted_gates = []
        remaining_gates = gate_names.copy()
        
        while remaining_gates:
            # Find gates with no unresolved dependencies
            ready_gates = []
            for gate_name in remaining_gates:
                gate = self.gates[gate_name]
                deps_satisfied = all(dep in sorted_gates or dep not in gate_names for dep in gate.depends_on)
                if deps_satisfied:
                    ready_gates.append(gate_name)
            
            if not ready_gates:
                # Circular dependency or missing dependency - add remaining gates anyway
                logger.warning("Possible circular dependency detected in quality gates")
                ready_gates = remaining_gates.copy()
            
            # Add ready gates to sorted list
            for gate_name in ready_gates:
                sorted_gates.append(gate_name)
                remaining_gates.remove(gate_name)
        
        return sorted_gates
    
    def _store_pipeline_run(self, run_id: str, pipeline_name: str, 
                          overall_status: str, results: List[QualityGateResult],
                          execution_time_ms: float, context: Dict[str, Any] = None) -> None:
        """Store pipeline run results in database."""
        with sqlite3.connect(self.db_path) as conn:
            # Store pipeline run
            conn.execute("""
                INSERT INTO pipeline_runs 
                (run_id, timestamp, pipeline_name, overall_status, total_gates, 
                 passed_gates, failed_gates, warning_gates, execution_time_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, datetime.now().isoformat(), pipeline_name, overall_status,
                len(results),
                sum(1 for r in results if r.status == GateStatus.PASSED),
                sum(1 for r in results if r.status == GateStatus.FAILED),
                sum(1 for r in results if r.status == GateStatus.WARNING),
                execution_time_ms,
                json.dumps(context) if context else None
            ))
            
            # Store individual gate results
            for result in results:
                conn.execute("""
                    INSERT INTO gate_results 
                    (timestamp, pipeline_run_id, gate_name, gate_type, status, severity, 
                     score, threshold, message, details, execution_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.timestamp, run_id, result.gate_name, result.gate_type,
                    result.status.value, result.severity.value, result.score,
                    result.threshold, result.message, json.dumps(result.details),
                    result.execution_time_ms
                ))
    
    def _generate_critical_failure_alerts(self, run_id: str, 
                                        critical_failures: List[QualityGateResult]) -> None:
        """Generate alerts for critical gate failures."""
        for failure in critical_failures:
            alert_data = {
                'alert_type': 'quality_gate_failure',
                'severity': 'critical',
                'pipeline_run_id': run_id,
                'gate_name': failure.gate_name,
                'gate_type': failure.gate_type,
                'score': failure.score,
                'threshold': failure.threshold,
                'message': failure.message,
                'timestamp': failure.timestamp
            }
            
            # Use alert manager to handle the alert
            try:
                self.alert_manager.check_alert_conditions(alert_data['score'], alert_data['threshold'])
                logger.info(f"Generated critical failure alert for gate: {failure.gate_name}")
            except Exception as e:
                logger.error(f"Error generating alert for gate {failure.gate_name}: {str(e)}")
    
    def get_pipeline_history(self, pipeline_name: str = None, 
                           days: int = 7) -> Dict[str, Any]:
        """
        Get pipeline execution history.
        
        Args:
            pipeline_name: Filter by pipeline name
            days: Number of days to look back
            
        Returns:
            Pipeline history data
        """
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM pipeline_runs WHERE timestamp >= ?"
            params = [cutoff_time]
            
            if pipeline_name:
                query += " AND pipeline_name = ?"
                params.append(pipeline_name)
            
            query += " ORDER BY timestamp DESC"
            
            cursor = conn.execute(query, params)
            runs = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            pipeline_runs = [dict(zip(columns, row)) for row in runs]
        
        # Calculate summary statistics
        total_runs = len(pipeline_runs)
        successful_runs = sum(1 for r in pipeline_runs if r['overall_status'] in ['passed', 'passed_with_warnings'])
        failed_runs = total_runs - successful_runs
        
        success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0
        
        # Recent trends
        recent_runs = pipeline_runs[:10]  # Last 10 runs
        avg_execution_time = sum(r['execution_time_ms'] for r in recent_runs) / len(recent_runs) if recent_runs else 0
        
        return {
            'generated_at': datetime.now().isoformat(),
            'time_window_days': days,
            'filter_pipeline': pipeline_name,
            'summary': {
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate_percent': success_rate,
                'avg_execution_time_ms': avg_execution_time
            },
            'recent_runs': recent_runs,
            'all_runs': pipeline_runs
        }
