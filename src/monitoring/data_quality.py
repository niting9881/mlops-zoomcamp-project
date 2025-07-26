"""
Data quality monitoring and validation module.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
# Skip Great Expectations for now - will use basic validation
# import great_expectations as ge

logger = logging.getLogger(__name__)


@dataclass
class DataQualityResult:
    """Data class for data quality check results."""
    timestamp: str
    dataset_name: str
    rule_name: str
    rule_type: str
    passed: bool
    severity: str
    expected_value: Any
    actual_value: Any
    message: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DataProfileResult:
    """Data class for data profiling results."""
    timestamp: str
    dataset_name: str
    total_rows: int
    total_columns: int
    missing_data_percent: float
    duplicate_rows_percent: float
    column_profiles: Dict[str, Any]
    data_types: Dict[str, str]
    summary_statistics: Dict[str, Any]


class DataQualityValidator:
    """Validates data quality using configurable rules."""
    
    def __init__(self, db_path: str = "data_quality.db"):
        """
        Initialize data quality validator.
        
        Args:
            db_path: Path to SQLite database for storing results
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"Data quality validator initialized with database: {db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for storing quality results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    severity TEXT NOT NULL,
                    expected_value TEXT,
                    actual_value TEXT,
                    message TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    total_rows INTEGER NOT NULL,
                    total_columns INTEGER NOT NULL,
                    missing_data_percent REAL NOT NULL,
                    duplicate_rows_percent REAL NOT NULL,
                    column_profiles TEXT NOT NULL,
                    data_types TEXT NOT NULL,
                    summary_statistics TEXT NOT NULL
                )
            """)
    
    def validate_completeness(self, df: pd.DataFrame, 
                            dataset_name: str,
                            required_columns: List[str] = None,
                            max_missing_percent: float = 5.0) -> List[DataQualityResult]:
        """
        Validate data completeness.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset
            required_columns: List of required columns
            max_missing_percent: Maximum allowed missing data percentage
            
        Returns:
            List of data quality results
        """
        results = []
        timestamp = datetime.now().isoformat()
        
        # Check required columns exist
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                results.append(DataQualityResult(
                    timestamp=timestamp,
                    dataset_name=dataset_name,
                    rule_name="required_columns_present",
                    rule_type="completeness",
                    passed=False,
                    severity="critical",
                    expected_value=required_columns,
                    actual_value=list(df.columns),
                    message=f"Missing required columns: {missing_columns}"
                ))
            else:
                results.append(DataQualityResult(
                    timestamp=timestamp,
                    dataset_name=dataset_name,
                    rule_name="required_columns_present",
                    rule_type="completeness",
                    passed=True,
                    severity="info",
                    expected_value=required_columns,
                    actual_value=list(df.columns),
                    message="All required columns are present"
                ))
        
        # Check missing data percentage per column
        for column in df.columns:
            missing_percent = (df[column].isnull().sum() / len(df)) * 100
            passed = missing_percent <= max_missing_percent
            severity = "critical" if missing_percent > 20 else "warning" if missing_percent > max_missing_percent else "info"
            
            results.append(DataQualityResult(
                timestamp=timestamp,
                dataset_name=dataset_name,
                rule_name=f"missing_data_{column}",
                rule_type="completeness",
                passed=passed,
                severity=severity,
                expected_value=f"<= {max_missing_percent}%",
                actual_value=f"{missing_percent:.2f}%",
                message=f"Column '{column}' has {missing_percent:.2f}% missing data"
            ))
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        empty_rows_percent = (empty_rows / len(df)) * 100 if len(df) > 0 else 0
        
        results.append(DataQualityResult(
            timestamp=timestamp,
            dataset_name=dataset_name,
            rule_name="empty_rows",
            rule_type="completeness",
            passed=empty_rows == 0,
            severity="warning" if empty_rows > 0 else "info",
            expected_value="0",
            actual_value=str(empty_rows),
            message=f"Found {empty_rows} completely empty rows ({empty_rows_percent:.2f}%)"
        ))
        
        return results
    
    def validate_consistency(self, df: pd.DataFrame, 
                           dataset_name: str,
                           consistency_rules: Dict[str, Any] = None) -> List[DataQualityResult]:
        """
        Validate data consistency.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset
            consistency_rules: Dictionary of consistency rules
            
        Returns:
            List of data quality results
        """
        results = []
        timestamp = datetime.now().isoformat()
        
        if consistency_rules is None:
            consistency_rules = {}
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percent = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        
        results.append(DataQualityResult(
            timestamp=timestamp,
            dataset_name=dataset_name,
            rule_name="duplicate_rows",
            rule_type="consistency",
            passed=duplicate_count == 0,
            severity="warning" if duplicate_count > 0 else "info",
            expected_value="0",
            actual_value=str(duplicate_count),
            message=f"Found {duplicate_count} duplicate rows ({duplicate_percent:.2f}%)"
        ))
        
        # Check data type consistency
        for column in df.columns:
            expected_type = consistency_rules.get(f"{column}_type")
            if expected_type:
                actual_type = str(df[column].dtype)
                passed = actual_type == expected_type
                
                results.append(DataQualityResult(
                    timestamp=timestamp,
                    dataset_name=dataset_name,
                    rule_name=f"data_type_{column}",
                    rule_type="consistency",
                    passed=passed,
                    severity="warning" if not passed else "info",
                    expected_value=expected_type,
                    actual_value=actual_type,
                    message=f"Column '{column}' type: expected {expected_type}, got {actual_type}"
                ))
        
        # Check value range consistency for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if f"{column}_min" in consistency_rules or f"{column}_max" in consistency_rules:
                min_val = consistency_rules.get(f"{column}_min")
                max_val = consistency_rules.get(f"{column}_max")
                
                actual_min = df[column].min()
                actual_max = df[column].max()
                
                if min_val is not None:
                    passed = actual_min >= min_val
                    results.append(DataQualityResult(
                        timestamp=timestamp,
                        dataset_name=dataset_name,
                        rule_name=f"min_value_{column}",
                        rule_type="consistency",
                        passed=passed,
                        severity="warning" if not passed else "info",
                        expected_value=f">= {min_val}",
                        actual_value=str(actual_min),
                        message=f"Column '{column}' minimum value check"
                    ))
                
                if max_val is not None:
                    passed = actual_max <= max_val
                    results.append(DataQualityResult(
                        timestamp=timestamp,
                        dataset_name=dataset_name,
                        rule_name=f"max_value_{column}",
                        rule_type="consistency",
                        passed=passed,
                        severity="warning" if not passed else "info",
                        expected_value=f"<= {max_val}",
                        actual_value=str(actual_max),
                        message=f"Column '{column}' maximum value check"
                    ))
        
        return results
    
    def validate_accuracy(self, df: pd.DataFrame, 
                         dataset_name: str,
                         reference_df: pd.DataFrame = None,
                         accuracy_rules: Dict[str, Any] = None) -> List[DataQualityResult]:
        """
        Validate data accuracy using statistical tests and reference data.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset
            reference_df: Reference dataset for comparison
            accuracy_rules: Dictionary of accuracy rules
            
        Returns:
            List of data quality results
        """
        results = []
        timestamp = datetime.now().isoformat()
        
        if accuracy_rules is None:
            accuracy_rules = {}
        
        # Outlier detection using IsolationForest
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            try:
                # Prepare data for outlier detection
                data_for_outliers = df[numeric_columns].dropna()
                if len(data_for_outliers) > 10:  # Need minimum samples for IsolationForest
                    
                    # Scale the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(data_for_outliers)
                    
                    # Detect outliers
                    contamination = accuracy_rules.get('outlier_contamination', 0.1)
                    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
                    outlier_labels = isolation_forest.fit_predict(scaled_data)
                    
                    outlier_count = np.sum(outlier_labels == -1)
                    outlier_percent = (outlier_count / len(data_for_outliers)) * 100
                    
                    # Consider high outlier percentage as potential accuracy issue
                    threshold = accuracy_rules.get('max_outlier_percent', 15.0)
                    passed = outlier_percent <= threshold
                    
                    results.append(DataQualityResult(
                        timestamp=timestamp,
                        dataset_name=dataset_name,
                        rule_name="outlier_detection",
                        rule_type="accuracy",
                        passed=passed,
                        severity="warning" if not passed else "info",
                        expected_value=f"<= {threshold}%",
                        actual_value=f"{outlier_percent:.2f}%",
                        message=f"Detected {outlier_count} outliers ({outlier_percent:.2f}% of data)",
                        metadata={'outlier_indices': np.where(outlier_labels == -1)[0].tolist()}
                    ))
                
            except Exception as e:
                logger.error(f"Error in outlier detection: {str(e)}")
                results.append(DataQualityResult(
                    timestamp=timestamp,
                    dataset_name=dataset_name,
                    rule_name="outlier_detection",
                    rule_type="accuracy",
                    passed=False,
                    severity="warning",
                    expected_value="successful_outlier_detection",
                    actual_value="error",
                    message=f"Outlier detection failed: {str(e)}"
                ))
        
        # Statistical comparison with reference data
        if reference_df is not None:
            for column in numeric_columns:
                if column in reference_df.columns:
                    try:
                        # Perform Kolmogorov-Smirnov test
                        current_data = df[column].dropna()
                        reference_data = reference_df[column].dropna()
                        
                        if len(current_data) > 0 and len(reference_data) > 0:
                            ks_statistic, p_value = stats.ks_2samp(current_data, reference_data)
                            
                            # Use p-value threshold to determine if distributions are significantly different
                            p_threshold = accuracy_rules.get('distribution_p_threshold', 0.05)
                            passed = p_value > p_threshold
                            
                            results.append(DataQualityResult(
                                timestamp=timestamp,
                                dataset_name=dataset_name,
                                rule_name=f"distribution_comparison_{column}",
                                rule_type="accuracy",
                                passed=passed,
                                severity="warning" if not passed else "info",
                                expected_value=f"p-value > {p_threshold}",
                                actual_value=f"{p_value:.4f}",
                                message=f"Distribution comparison for '{column}': KS statistic={ks_statistic:.4f}, p-value={p_value:.4f}",
                                metadata={
                                    'ks_statistic': ks_statistic,
                                    'p_value': p_value,
                                    'current_mean': float(current_data.mean()),
                                    'reference_mean': float(reference_data.mean()),
                                    'current_std': float(current_data.std()),
                                    'reference_std': float(reference_data.std())
                                }
                            ))
                    
                    except Exception as e:
                        logger.error(f"Error in distribution comparison for {column}: {str(e)}")
        
        return results
    
    def create_data_profile(self, df: pd.DataFrame, dataset_name: str) -> DataProfileResult:
        """
        Create comprehensive data profile.
        
        Args:
            df: DataFrame to profile
            dataset_name: Name of the dataset
            
        Returns:
            Data profile result
        """
        timestamp = datetime.now().isoformat()
        
        # Basic statistics
        total_rows = len(df)
        total_columns = len(df.columns)
        missing_data_percent = (df.isnull().sum().sum() / (total_rows * total_columns)) * 100 if total_rows > 0 else 0
        duplicate_rows_percent = (df.duplicated().sum() / total_rows) * 100 if total_rows > 0 else 0
        
        # Column-wise profiles
        column_profiles = {}
        for column in df.columns:
            col_data = df[column]
            
            profile = {
                'data_type': str(col_data.dtype),
                'non_null_count': int(col_data.count()),
                'null_count': int(col_data.isnull().sum()),
                'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100 if len(col_data) > 0 else 0,
                'unique_count': int(col_data.nunique()),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100 if len(col_data) > 0 else 0
            }
            
            # Numeric column statistics
            if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_data = col_data.dropna()
                if len(numeric_data) > 0:
                    profile.update({
                        'min': float(numeric_data.min()),
                        'max': float(numeric_data.max()),
                        'mean': float(numeric_data.mean()),
                        'median': float(numeric_data.median()),
                        'std': float(numeric_data.std()),
                        'q25': float(numeric_data.quantile(0.25)),
                        'q75': float(numeric_data.quantile(0.75)),
                        'skewness': float(numeric_data.skew()),
                        'kurtosis': float(numeric_data.kurtosis())
                    })
            
            # Categorical column statistics
            elif col_data.dtype == 'object' or col_data.dtype.name == 'category':
                value_counts = col_data.value_counts()
                if len(value_counts) > 0:
                    profile.update({
                        'top_value': str(value_counts.index[0]),
                        'top_value_frequency': int(value_counts.iloc[0]),
                        'top_value_percentage': (value_counts.iloc[0] / len(col_data)) * 100,
                        'value_counts': value_counts.head(10).to_dict()  # Top 10 values
                    })
            
            column_profiles[column] = profile
        
        # Data types summary
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Overall summary statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        summary_statistics = {
            'numeric_columns_count': len(numeric_columns),
            'categorical_columns_count': len(df.columns) - len(numeric_columns),
            'total_missing_values': int(df.isnull().sum().sum()),
            'completely_empty_rows': int(df.isnull().all(axis=1).sum()),
            'completely_empty_columns': int(df.isnull().all(axis=0).sum())
        }
        
        if len(numeric_columns) > 0:
            numeric_df = df[numeric_columns]
            summary_statistics.update({
                'numeric_correlations': numeric_df.corr().to_dict(),
                'numeric_summary': numeric_df.describe().to_dict()
            })
        
        return DataProfileResult(
            timestamp=timestamp,
            dataset_name=dataset_name,
            total_rows=total_rows,
            total_columns=total_columns,
            missing_data_percent=missing_data_percent,
            duplicate_rows_percent=duplicate_rows_percent,
            column_profiles=column_profiles,
            data_types=data_types,
            summary_statistics=summary_statistics
        )
    
    def store_results(self, results: List[DataQualityResult]) -> None:
        """
        Store quality results in database.
        
        Args:
            results: List of quality results to store
        """
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO quality_results 
                    (timestamp, dataset_name, rule_name, rule_type, passed, severity, 
                     expected_value, actual_value, message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.timestamp, result.dataset_name, result.rule_name, result.rule_type,
                    result.passed, result.severity, str(result.expected_value), 
                    str(result.actual_value), result.message, 
                    json.dumps(result.metadata) if result.metadata else None
                ))
    
    def store_profile(self, profile: DataProfileResult) -> None:
        """
        Store data profile in database.
        
        Args:
            profile: Data profile to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_profiles 
                (timestamp, dataset_name, total_rows, total_columns, missing_data_percent, 
                 duplicate_rows_percent, column_profiles, data_types, summary_statistics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.timestamp, profile.dataset_name, profile.total_rows, profile.total_columns,
                profile.missing_data_percent, profile.duplicate_rows_percent,
                json.dumps(profile.column_profiles), json.dumps(profile.data_types),
                json.dumps(profile.summary_statistics)
            ))
    
    def get_quality_report(self, dataset_name: str = None, 
                          hours: int = 24) -> Dict[str, Any]:
        """
        Get quality report for specified time window.
        
        Args:
            dataset_name: Filter by dataset name
            hours: Time window in hours
            
        Returns:
            Quality report dictionary
        """
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Build query
            query = "SELECT * FROM quality_results WHERE timestamp >= ?"
            params = [cutoff_time]
            
            if dataset_name:
                query += " AND dataset_name = ?"
                params.append(dataset_name)
            
            query += " ORDER BY timestamp DESC"
            
            cursor = conn.execute(query, params)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            quality_results = [dict(zip(columns, row)) for row in results]
        
        # Aggregate statistics
        total_checks = len(quality_results)
        passed_checks = sum(1 for r in quality_results if r['passed'])
        failed_checks = total_checks - passed_checks
        
        # Group by severity
        severity_counts = {}
        rule_type_counts = {}
        dataset_counts = {}
        
        for result in quality_results:
            severity = result['severity']
            rule_type = result['rule_type']
            dataset = result['dataset_name']
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            rule_type_counts[rule_type] = rule_type_counts.get(rule_type, 0) + 1
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        # Recent failures
        recent_failures = [r for r in quality_results if not r['passed']][:10]  # Top 10 recent failures
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'time_window_hours': hours,
            'filter_dataset': dataset_name,
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'pass_rate_percent': (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            },
            'breakdown': {
                'by_severity': severity_counts,
                'by_rule_type': rule_type_counts,
                'by_dataset': dataset_counts
            },
            'recent_failures': recent_failures,
            'all_results': quality_results
        }
        
        return report
    
    def run_comprehensive_validation(self, df: pd.DataFrame, 
                                   dataset_name: str,
                                   reference_df: pd.DataFrame = None,
                                   validation_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset
            reference_df: Reference dataset for comparison
            validation_config: Validation configuration
            
        Returns:
            Comprehensive validation results
        """
        if validation_config is None:
            validation_config = {}
        
        logger.info(f"Starting comprehensive validation for dataset: {dataset_name}")
        
        all_results = []
        
        # Run completeness validation
        completeness_results = self.validate_completeness(
            df, dataset_name,
            required_columns=validation_config.get('required_columns'),
            max_missing_percent=validation_config.get('max_missing_percent', 5.0)
        )
        all_results.extend(completeness_results)
        
        # Run consistency validation
        consistency_results = self.validate_consistency(
            df, dataset_name,
            consistency_rules=validation_config.get('consistency_rules', {})
        )
        all_results.extend(consistency_results)
        
        # Run accuracy validation
        accuracy_results = self.validate_accuracy(
            df, dataset_name, reference_df,
            accuracy_rules=validation_config.get('accuracy_rules', {})
        )
        all_results.extend(accuracy_results)
        
        # Create data profile
        profile = self.create_data_profile(df, dataset_name)
        
        # Store results
        self.store_results(all_results)
        self.store_profile(profile)
        
        # Calculate summary
        total_checks = len(all_results)
        passed_checks = sum(1 for r in all_results if r.passed)
        critical_failures = sum(1 for r in all_results if not r.passed and r.severity == 'critical')
        warning_failures = sum(1 for r in all_results if not r.passed and r.severity == 'warning')
        
        validation_summary = {
            'dataset_name': dataset_name,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'passed' if critical_failures == 0 else 'failed',
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': total_checks - passed_checks,
                'critical_failures': critical_failures,
                'warning_failures': warning_failures,
                'pass_rate_percent': (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            },
            'data_profile': profile,
            'detailed_results': all_results
        }
        
        logger.info(f"Validation completed for {dataset_name}: {passed_checks}/{total_checks} checks passed")
        
        return validation_summary


class GreatExpectationsIntegration:
    """Integration with Great Expectations for advanced data validation."""
    
    def __init__(self, context_root_dir: str = "great_expectations"):
        """
        Initialize Great Expectations integration.
        
        Args:
            context_root_dir: Root directory for GE context
        """
        self.context_root_dir = Path(context_root_dir)
        self.context = None
        logger.info(f"Great Expectations integration initialized with context dir: {context_root_dir}")
    
    def create_expectation_suite(self, df: pd.DataFrame, 
                               suite_name: str) -> Dict[str, Any]:
        """
        Create expectation suite based on data profiling.
        
        Args:
            df: DataFrame to create expectations for
            suite_name: Name of the expectation suite
            
        Returns:
            Expectation suite dictionary
        """
        try:
            # Simplified expectation suite creation without deprecated PandasDataset
            expectations = []
            
            # Create basic expectations for each column
            for column in df.columns:
                col_data = df[column]
                
                # Expect column to exist
                expectations.append({
                    'expectation_type': 'expect_column_to_exist',
                    'kwargs': {'column': column}
                })
                
                # For numeric columns
                if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_data = col_data.dropna()
                    if len(numeric_data) > 0:
                        # Expect values to be between min and max (with some buffer)
                        min_val = numeric_data.min()
                        max_val = numeric_data.max()
                        buffer = (max_val - min_val) * 0.1  # 10% buffer
                        
                        expectations.append({
                            'expectation_type': 'expect_column_values_to_be_between',
                            'kwargs': {
                                'column': column,
                                'min_value': min_val - buffer,
                                'max_value': max_val + buffer,
                                'mostly': 0.95  # 95% of values should be within range
                            }
                        })
                        
                        # Expect column to not be null (if low null rate)
                        null_rate = col_data.isnull().sum() / len(col_data)
                        if null_rate < 0.05:  # Less than 5% nulls
                            expectations.append({
                                'expectation_type': 'expect_column_values_to_not_be_null',
                                'kwargs': {'column': column, 'mostly': 0.95}
                            })
                
                # For categorical columns
                elif col_data.dtype == 'object':
                    # Expect column values to be in set (top values)
                    value_counts = col_data.value_counts()
                    if len(value_counts) <= 50:  # Only for columns with limited unique values
                        top_values = value_counts.head(20).index.tolist()  # Top 20 values
                        expectations.append({
                            'expectation_type': 'expect_column_values_to_be_in_set',
                            'kwargs': {
                                'column': column,
                                'value_set': top_values,
                                'mostly': 0.90  # 90% should be in the expected set
                            }
                        })
            
            # Table-level expectations
            expectations.append({
                'expectation_type': 'expect_table_row_count_to_be_between',
                'kwargs': {
                    'min_value': int(len(df) * 0.8),  # Allow 20% deviation
                    'max_value': int(len(df) * 1.2)
                }
            })
            
            return {
                'expectation_suite_name': suite_name,
                'expectations': expectations,
                'meta': {
                    'created_at': datetime.now().isoformat(),
                    'source_dataset_shape': df.shape,
                    'source_columns': list(df.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating expectation suite: {str(e)}")
            return {'error': str(e)}
    
    def validate_with_expectations(self, df: pd.DataFrame, 
                                 expectation_suite: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate DataFrame against expectation suite.
        
        Args:
            df: DataFrame to validate
            expectation_suite: Expectation suite dictionary
            
        Returns:
            Validation results
        """
        try:
            # Simplified validation without deprecated PandasDataset
            results = []
            
            for expectation in expectation_suite.get('expectations', []):
                expectation_type = expectation['expectation_type']
                kwargs = expectation['kwargs']
                
                try:
                    # Manual validation for common expectation types
                    success = self._validate_expectation(df, expectation_type, kwargs)
                    results.append({
                        'expectation_type': expectation_type,
                        'kwargs': kwargs,
                        'success': success,
                        'result': {'success': success}
                    })
                        
                except Exception as e:
                    results.append({
                        'expectation_type': expectation_type,
                        'kwargs': kwargs,
                        'success': False,
                        'error': str(e)
                    })
            
            # Calculate overall success
            successful_expectations = sum(1 for r in results if r['success'])
            total_expectations = len(results)
            success_percentage = (successful_expectations / total_expectations) * 100 if total_expectations > 0 else 0
            
            return {
                'validation_timestamp': datetime.now().isoformat(),
                'expectation_suite_name': expectation_suite.get('expectation_suite_name'),
                'success': success_percentage == 100,
                'success_percentage': success_percentage,
                'successful_expectations': successful_expectations,
                'total_expectations': total_expectations,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error validating with expectations: {str(e)}")
            return {
                'validation_timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def _validate_expectation(self, df: pd.DataFrame, expectation_type: str, kwargs: Dict[str, Any]) -> bool:
        """
        Manual validation for common expectation types.
        
        Args:
            df: DataFrame to validate
            expectation_type: Type of expectation
            kwargs: Expectation parameters
            
        Returns:
            True if expectation is met, False otherwise
        """
        try:
            column = kwargs.get('column')
            mostly = kwargs.get('mostly', 1.0)  # Default to 100%
            
            if expectation_type == 'expect_column_to_exist':
                return column in df.columns
                
            elif expectation_type == 'expect_column_values_to_not_be_null':
                if column not in df.columns:
                    return False
                non_null_rate = df[column].notna().sum() / len(df)
                return non_null_rate >= mostly
                
            elif expectation_type == 'expect_column_values_to_be_between':
                if column not in df.columns:
                    return False
                min_value = kwargs.get('min_value')
                max_value = kwargs.get('max_value')
                if min_value is None or max_value is None:
                    return False
                valid_values = df[column].dropna()
                in_range_rate = ((valid_values >= min_value) & (valid_values <= max_value)).sum() / len(valid_values)
                return in_range_rate >= mostly
                
            elif expectation_type == 'expect_column_values_to_be_in_set':
                if column not in df.columns:
                    return False
                value_set = set(kwargs.get('value_set', []))
                if not value_set:
                    return False
                valid_values = df[column].dropna()
                in_set_rate = valid_values.isin(value_set).sum() / len(valid_values)
                return in_set_rate >= mostly
                
            elif expectation_type == 'expect_table_row_count_to_be_between':
                min_value = kwargs.get('min_value', 0)
                max_value = kwargs.get('max_value', float('inf'))
                row_count = len(df)
                return min_value <= row_count <= max_value
                
            else:
                # Unknown expectation type - return True to avoid failing
                logger.warning(f"Unknown expectation type: {expectation_type}")
                return True
                
        except Exception as e:
            logger.error(f"Error validating expectation {expectation_type}: {str(e)}")
            return False
