"""
Example usage and integration script for Phase 2 MLOps Monitoring & Quality features.

This script demonstrates how to use the comprehensive monitoring system for:
- Data quality validation
- Model performance monitoring
- Drift detection
- Quality gates execution
- Dashboard generation
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from monitoring import (
    DataDriftDetector,
    ModelPerformanceMonitor,
    DataQualityValidator,
    QualityGateEngine,
    MonitoringDashboard,
    MetricsTracker,
    APIMetricsCollector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data():
    """Load sample house price data for demonstration."""
    try:
        # Try to load real data
        data_path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_house_data.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            logger.info(f"Loaded real data with shape: {df.shape}")
            return df
        else:
            # Generate synthetic data for demonstration
            logger.info("Real data not found, generating synthetic data")
            np.random.seed(42)
            n_samples = 1000
            
            df = pd.DataFrame({
                'price': np.random.normal(500000, 150000, n_samples),
                'bedrooms': np.random.randint(1, 6, n_samples),
                'bathrooms': np.random.uniform(1, 4, n_samples),
                'sqft_living': np.random.normal(2000, 500, n_samples),
                'sqft_lot': np.random.normal(8000, 3000, n_samples),
                'floors': np.random.randint(1, 4, n_samples),
                'waterfront': np.random.binomial(1, 0.1, n_samples),
                'view': np.random.randint(0, 5, n_samples),
                'condition': np.random.randint(1, 6, n_samples),
                'grade': np.random.randint(3, 14, n_samples),
                'yr_built': np.random.randint(1900, 2021, n_samples),
                'yr_renovated': np.random.choice([0] + list(range(1950, 2021)), n_samples),
                'zipcode': np.random.choice(['98001', '98002', '98003', '98004', '98005'], n_samples),
                'lat': np.random.uniform(47.1, 47.8, n_samples),
                'long': np.random.uniform(-122.5, -121.3, n_samples)
            })
            
            # Ensure positive values
            df['price'] = np.abs(df['price'])
            df['sqft_living'] = np.abs(df['sqft_living'])
            df['sqft_lot'] = np.abs(df['sqft_lot'])
            
            logger.info(f"Generated synthetic data with shape: {df.shape}")
            return df
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def demonstrate_data_quality_validation():
    """Demonstrate comprehensive data quality validation."""
    logger.info("=== Data Quality Validation Demo ===")
    
    # Load data
    df = load_sample_data()
    
    # Initialize data quality validator
    validator = DataQualityValidator(db_path="demo_data_quality.db")
    
    # Define validation configuration
    validation_config = {
        'required_columns': ['price', 'bedrooms', 'bathrooms', 'sqft_living'],
        'max_missing_percent': 5.0,
        'consistency_rules': {
            'price_min': 50000,
            'price_max': 10000000,
            'bedrooms_min': 1,
            'bedrooms_max': 10,
            'sqft_living_min': 500,
            'sqft_living_max': 20000
        },
        'accuracy_rules': {
            'outlier_contamination': 0.1,
            'max_outlier_percent': 15.0
        }
    }
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation(
        df, 
        "house_price_data",
        validation_config=validation_config
    )
    
    # Display results
    logger.info(f"Data Quality Results:")
    logger.info(f"  Overall Status: {results['overall_status']}")
    logger.info(f"  Total Checks: {results['summary']['total_checks']}")
    logger.info(f"  Passed: {results['summary']['passed_checks']}")
    logger.info(f"  Failed: {results['summary']['failed_checks']}")
    logger.info(f"  Pass Rate: {results['summary']['pass_rate_percent']:.1f}%")
    
    # Show some detailed results
    logger.info("\nDetailed Results (first 5):")
    for i, result in enumerate(results['detailed_results'][:5]):
        logger.info(f"  {i+1}. {result.rule_name}: {'PASS' if result.passed else 'FAIL'} - {result.message}")
    
    return results


def demonstrate_drift_detection():
    """Demonstrate data drift detection."""
    logger.info("\n=== Data Drift Detection Demo ===")
    
    # Load reference and current data
    reference_data = load_sample_data()
    
    # Create "drifted" data by shifting some distributions
    current_data = reference_data.copy()
    current_data['price'] = current_data['price'] * 1.2  # 20% price increase
    current_data['sqft_living'] = current_data['sqft_living'] + np.random.normal(0, 100, len(current_data))
    
    # Initialize drift detector
    drift_detector = DataDriftDetector()
    
    # Fit on reference data
    drift_detector.fit_reference(reference_data)
    
    # Detect drift
    drift_results = drift_detector.detect_drift(current_data)
    
    # Display results
    logger.info("Drift Detection Results:")
    for column, result in drift_results.items():
        drift_detected = result.get('p_value', 1.0) < 0.05
        logger.info(f"  {column}: {'DRIFT DETECTED' if drift_detected else 'NO DRIFT'} "
                   f"(p-value: {result.get('p_value', 'N/A'):.4f})")
    
    return drift_results


def demonstrate_model_performance_monitoring():
    """Demonstrate model performance monitoring."""
    logger.info("\n=== Model Performance Monitoring Demo ===")
    
    # Generate synthetic model predictions
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate true values and predictions
    y_true = np.random.normal(500000, 150000, n_samples)
    y_pred = y_true + np.random.normal(0, 50000, n_samples)  # Add some prediction error
    
    # Initialize performance monitor
    performance_monitor = ModelPerformanceMonitor()
    
    # Evaluate performance
    results = performance_monitor.evaluate_batch(y_true, y_pred)
    
    # Display results
    logger.info("Model Performance Results:")
    for metric, value in results['metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"\nPerformance Status: {results['performance_status']}")
    logger.info(f"Baseline Comparison: {'Better' if results['baseline_comparison']['performance_change'] > 0 else 'Worse'}")
    
    return results


def demonstrate_quality_gates():
    """Demonstrate quality gates execution."""
    logger.info("\n=== Quality Gates Demo ===")
    
    # Initialize quality gate engine
    gate_engine = QualityGateEngine(db_path="demo_quality_gates.db")
    
    # Create default configuration
    config_path = "demo_quality_gates_config.yaml"
    gate_engine.create_default_configuration(config_path)
    gate_engine.load_configuration(config_path)
    
    # Prepare data for gates
    df = load_sample_data()
    reference_data = df.sample(n=500, random_state=42)
    current_data = df.sample(n=500, random_state=123)
    
    # Generate synthetic predictions
    y_true = current_data['price'].values
    y_pred = y_true + np.random.normal(0, 50000, len(y_true))
    
    # Prepare gate data
    gate_data = {
        'dataframe': current_data,
        'dataset_name': 'house_price_validation',
        'current_data': current_data,
        'reference_data': reference_data,
        'y_true': y_true,
        'y_pred': y_pred,
        'predictions': y_pred,
        'feature_importance': {
            'sqft_living': 0.3,
            'bedrooms': 0.2,
            'bathrooms': 0.15,
            'grade': 0.1,
            'condition': 0.08,
            'floors': 0.05,
            'view': 0.05,
            'waterfront': 0.04,
            'yr_built': 0.03
        }
    }
    
    # Execute pipeline
    pipeline_results = gate_engine.execute_pipeline(
        pipeline_name="house_price_validation",
        data=gate_data,
        context={'expected_rows': 500}
    )
    
    # Display results
    logger.info("Quality Gates Pipeline Results:")
    logger.info(f"  Run ID: {pipeline_results['run_id']}")
    logger.info(f"  Overall Status: {pipeline_results['overall_status']}")
    logger.info(f"  Total Gates: {pipeline_results['summary']['total_gates']}")
    logger.info(f"  Passed: {pipeline_results['summary']['passed_gates']}")
    logger.info(f"  Failed: {pipeline_results['summary']['failed_gates']}")
    logger.info(f"  Warnings: {pipeline_results['summary']['warning_gates']}")
    
    # Show gate results
    logger.info("\nIndividual Gate Results:")
    for gate_result in pipeline_results['gate_results'][:5]:  # Show first 5
        logger.info(f"  {gate_result['gate_name']}: {gate_result['status']} "
                   f"(score: {gate_result['score']:.3f}, threshold: {gate_result['threshold']})")
    
    return pipeline_results


def demonstrate_monitoring_dashboard():
    """Demonstrate monitoring dashboard generation."""
    logger.info("\n=== Monitoring Dashboard Demo ===")
    
    # Initialize dashboard
    dashboard = MonitoringDashboard(
        db_path="demo_monitoring.db",
        quality_gates_db="demo_quality_gates.db",
        data_quality_db="demo_data_quality.db"
    )
    
    # Generate overview dashboard
    overview = dashboard.generate_overview_dashboard(hours=24)
    
    logger.info("Dashboard Overview:")
    logger.info(f"  Overall Health Score: {overview['overall_health']['score']}")
    logger.info(f"  Overall Status: {overview['overall_health']['status']}")
    
    # Show key metrics
    key_metrics = overview.get('key_metrics', {})
    logger.info("\nKey Metrics:")
    for metric_name, value in key_metrics.items():
        logger.info(f"  {metric_name}: {value}")
    
    # Generate specific dashboards
    logger.info("\nGenerating specific dashboards...")
    
    try:
        quality_dashboard = dashboard.generate_data_quality_dashboard(days=7)
        logger.info(f"Data Quality Dashboard generated for last 7 days")
    except Exception as e:
        logger.warning(f"Could not generate data quality dashboard: {str(e)}")
    
    try:
        performance_dashboard = dashboard.generate_model_performance_dashboard(days=7)
        logger.info(f"Model Performance Dashboard generated for last 7 days")
    except Exception as e:
        logger.warning(f"Could not generate performance dashboard: {str(e)}")
    
    # Export dashboard report
    try:
        dashboard.export_dashboard_report('overview', 'monitoring_report.json', 'json')
        logger.info("Dashboard report exported to monitoring_report.json")
        
        dashboard.export_dashboard_report('overview', 'monitoring_report.html', 'html')
        logger.info("Dashboard report exported to monitoring_report.html")
    except Exception as e:
        logger.warning(f"Could not export dashboard report: {str(e)}")
    
    return overview


def demonstrate_integration_scenario():
    """Demonstrate complete integration scenario."""
    logger.info("\n=== Complete Integration Scenario ===")
    
    try:
        # Step 1: Data Quality Validation
        logger.info("Step 1: Running data quality validation...")
        quality_results = demonstrate_data_quality_validation()
        
        # Step 2: Drift Detection
        logger.info("\nStep 2: Running drift detection...")
        drift_results = demonstrate_drift_detection()
        
        # Step 3: Model Performance Monitoring
        logger.info("\nStep 3: Running model performance monitoring...")
        performance_results = demonstrate_model_performance_monitoring()
        
        # Step 4: Quality Gates Execution
        logger.info("\nStep 4: Running quality gates...")
        gates_results = demonstrate_quality_gates()
        
        # Step 5: Dashboard Generation
        logger.info("\nStep 5: Generating monitoring dashboard...")
        dashboard_results = demonstrate_monitoring_dashboard()
        
        # Integration Summary
        logger.info("\n=== Integration Summary ===")
        logger.info(f"✅ Data Quality: {quality_results['overall_status']}")
        logger.info(f"✅ Drift Detection: {'Some drift detected' if any(r.get('p_value', 1) < 0.05 for r in drift_results.values()) else 'No drift detected'}")
        logger.info(f"✅ Model Performance: {performance_results['performance_status']}")
        logger.info(f"✅ Quality Gates: {gates_results['overall_status']}")
        logger.info(f"✅ Dashboard: Generated with health score {dashboard_results['overall_health']['score']}")
        
        logger.info("\n🎉 Phase 2 MLOps Monitoring & Quality features successfully demonstrated!")
        
        return {
            'data_quality': quality_results,
            'drift_detection': drift_results,
            'model_performance': performance_results,
            'quality_gates': gates_results,
            'dashboard': dashboard_results
        }
        
    except Exception as e:
        logger.error(f"Integration scenario failed: {str(e)}")
        raise


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Phase 2 MLOps Monitoring & Quality Demo")
    logger.info("=" * 60)
    
    try:
        # Run complete integration scenario
        results = demonstrate_integration_scenario()
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 Demo completed successfully!")
        logger.info("📁 Check the generated files:")
        logger.info("   - demo_data_quality.db (Data quality results)")
        logger.info("   - demo_quality_gates.db (Quality gates results)")
        logger.info("   - demo_monitoring.db (Dashboard data)")
        logger.info("   - monitoring_report.json (Dashboard export)")
        logger.info("   - monitoring_report.html (Dashboard HTML)")
        logger.info("   - demo_quality_gates_config.yaml (Quality gates config)")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
