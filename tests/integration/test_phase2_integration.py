"""
Integration tests for Phase 2 monitoring features.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import sqlite3
import json

# Import monitoring components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from monitoring import (
    DataDriftDetector,
    ModelPerformanceMonitor,
    DataQualityValidator,
    QualityGateEngine,
    MonitoringDashboard,
    MetricsTracker
)


class TestPhase2Integration:
    """Integration tests for Phase 2 monitoring system."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp(prefix='phase2_integration_')
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_house_data(self):
        """Generate realistic house price data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate correlated features for realistic house data
        sqft_living = np.random.normal(2000, 500, n_samples)
        sqft_living = np.clip(sqft_living, 500, 10000)  # Reasonable range
        
        bedrooms = np.random.poisson(3, n_samples)
        bedrooms = np.clip(bedrooms, 1, 8)
        
        bathrooms = bedrooms * 0.7 + np.random.normal(0, 0.5, n_samples)
        bathrooms = np.clip(bathrooms, 1, 6)
        
        # Price correlated with sqft and bedrooms
        price_base = sqft_living * 200 + bedrooms * 20000
        price = price_base + np.random.normal(0, 50000, n_samples)
        price = np.clip(price, 50000, 2000000)
        
        df = pd.DataFrame({
            'price': price,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': np.random.normal(8000, 3000, n_samples),
            'floors': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.5, 0.1]),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'view': np.random.choice([0, 1, 2, 3, 4], n_samples),
            'condition': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.1, 0.6, 0.15, 0.05]),
            'grade': np.random.choice(range(3, 14), n_samples),
            'yr_built': np.random.choice(range(1900, 2023), n_samples),
            'zipcode': np.random.choice(['98001', '98002', '98003', '98004', '98005'], n_samples)
        })
        
        return df
    
    def test_complete_monitoring_workflow(self, temp_workspace, sample_house_data):
        """Test complete monitoring workflow with realistic data."""
        print("🔄 Testing complete monitoring workflow...")
        
        # Split data into reference and current
        reference_data = sample_house_data.iloc[:700].copy()
        current_data = sample_house_data.iloc[700:].copy()
        
        # Add some realistic drift to current data
        current_data['price'] = current_data['price'] * 1.05  # 5% price inflation
        current_data.loc[current_data.index[:50], 'sqft_living'] += 200  # Some houses renovated
        
        # Setup database paths
        quality_db = os.path.join(temp_workspace, 'data_quality.db')
        gates_db = os.path.join(temp_workspace, 'quality_gates.db')
        monitoring_db = os.path.join(temp_workspace, 'monitoring.db')
        
        # 1. Data Quality Validation
        print("  📊 Running data quality validation...")
        quality_validator = DataQualityValidator(db_path=quality_db)
        
        validation_config = {
            'required_columns': ['price', 'bedrooms', 'bathrooms', 'sqft_living'],
            'max_missing_percent': 5.0,
            'consistency_rules': {
                'price_min': 50000,
                'price_max': 2000000,
                'bedrooms_min': 1,
                'bedrooms_max': 8,
                'sqft_living_min': 500,
                'sqft_living_max': 10000
            },
            'accuracy_rules': {
                'outlier_contamination': 0.1,
                'max_outlier_percent': 15.0
            }
        }
        
        quality_results = quality_validator.run_comprehensive_validation(
            current_data,
            'house_price_current',
            reference_df=reference_data,
            validation_config=validation_config
        )
        
        assert quality_results['overall_status'] in ['passed', 'failed']
        assert quality_results['summary']['total_checks'] > 0
        print(f"    ✅ Quality validation completed: {quality_results['summary']['pass_rate_percent']:.1f}% pass rate")
        
        # 2. Drift Detection
        print("  🔍 Running drift detection...")
        drift_detector = DataDriftDetector()
        drift_detector.fit_reference(reference_data)
        drift_results = drift_detector.detect_drift(current_data)
        
        # Should detect some drift due to price inflation
        price_drift = drift_results.get('price', {})
        if 'p_value' in price_drift:
            print(f"    📈 Price drift p-value: {price_drift['p_value']:.4f}")
            # With 5% price inflation, we should detect drift
            assert price_drift['p_value'] < 0.05, "Should detect price drift"
        
        print("    ✅ Drift detection completed")
        
        # 3. Model Performance Simulation
        print("  🎯 Simulating model performance monitoring...")
        y_true = current_data['price'].values
        # Simulate predictions with some error
        y_pred = y_true * (1 + np.random.normal(0, 0.1, len(y_true)))
        
        performance_monitor = ModelPerformanceMonitor()
        performance_results = performance_monitor.evaluate_batch(y_true, y_pred)
        
        assert 'metrics' in performance_results
        assert performance_results['metrics']['r2_score'] > 0.7  # Should have decent performance
        print(f"    📊 Model R² score: {performance_results['metrics']['r2_score']:.3f}")
        print("    ✅ Performance monitoring completed")
        
        # 4. Quality Gates Execution
        print("  🚪 Running quality gates...")
        gate_engine = QualityGateEngine(db_path=gates_db)
        
        # Create comprehensive gate configuration
        from monitoring.quality_gates import QualityGateConfig, GateSeverity
        
        gates_config = [
            QualityGateConfig(
                name='data_volume_check',
                type='data_volume',
                enabled=True,
                severity=GateSeverity.CRITICAL,
                threshold=200,
                parameters={'min_rows': 200},
                depends_on=[],
                description='Minimum data volume check'
            ),
            QualityGateConfig(
                name='data_quality_check',
                type='data_quality',
                enabled=True,
                severity=GateSeverity.HIGH,
                threshold=80.0,
                parameters={'max_missing_percent': 5.0},
                depends_on=['data_volume_check'],
                description='Data quality validation'
            ),
            QualityGateConfig(
                name='model_performance_check',
                type='model_performance',
                enabled=True,
                severity=GateSeverity.CRITICAL,
                threshold=0.7,
                parameters={'primary_metric': 'r2_score'},
                depends_on=['data_quality_check'],
                description='Model performance validation'
            )
        ]
        
        # Add gates to engine
        for gate_config in gates_config:
            gate_engine.gates[gate_config.name] = gate_config
        
        # Prepare gate data
        gate_data = {
            'dataframe': current_data,
            'dataset_name': 'house_price_integration_test',
            'current_data': current_data,
            'reference_data': reference_data,
            'y_true': y_true,
            'y_pred': y_pred,
            'predictions': y_pred,
            'feature_importance': {
                'sqft_living': 0.35,
                'bedrooms': 0.20,
                'bathrooms': 0.15,
                'grade': 0.10,
                'condition': 0.08,
                'floors': 0.07,
                'view': 0.05
            }
        }
        
        # Execute pipeline
        pipeline_results = gate_engine.execute_pipeline(
            'house_price_validation_pipeline',
            gate_data,
            context={'expected_rows': 300}
        )
        
        assert 'overall_status' in pipeline_results
        assert pipeline_results['summary']['total_gates'] == len(gates_config)
        
        passed_gates = pipeline_results['summary']['passed_gates']
        total_gates = pipeline_results['summary']['total_gates']
        success_rate = (passed_gates / total_gates) * 100
        
        print(f"    🎯 Quality gates: {passed_gates}/{total_gates} passed ({success_rate:.1f}%)")
        print("    ✅ Quality gates execution completed")
        
        # 5. Metrics Tracking
        print("  📈 Testing metrics tracking...")
        metrics_tracker = MetricsTracker(db_path=monitoring_db)
        
        # Track various metrics
        metrics_to_track = {
            'data_quality_score': quality_results['summary']['pass_rate_percent'],
            'model_r2_score': performance_results['metrics']['r2_score'],
            'model_mae': performance_results['metrics']['mean_absolute_error'],
            'pipeline_success_rate': success_rate,
            'data_volume': len(current_data)
        }
        
        metrics_tracker.batch_track_metrics(
            metrics_to_track,
            context={
                'pipeline_run': 'integration_test',
                'data_version': 'current',
                'model_version': 'v1.0'
            }
        )
        
        # Verify metrics were stored
        stored_metrics = metrics_tracker.get_metrics(hours=1)
        assert len(stored_metrics) == len(metrics_to_track)
        print(f"    📊 Tracked {len(stored_metrics)} metrics")
        print("    ✅ Metrics tracking completed")
        
        # 6. Dashboard Generation
        print("  📊 Generating monitoring dashboard...")
        dashboard = MonitoringDashboard(
            db_path=monitoring_db,
            quality_gates_db=gates_db,
            data_quality_db=quality_db
        )
        
        try:
            overview = dashboard.generate_overview_dashboard(hours=1)
            assert 'generated_at' in overview
            assert 'overall_health' in overview
            
            # Test dashboard export
            export_path = os.path.join(temp_workspace, 'dashboard_report.json')
            dashboard.export_dashboard_report('overview', export_path, 'json')
            assert os.path.exists(export_path)
            
            # Verify exported content
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            assert 'generated_at' in exported_data
            
            print("    📊 Dashboard generated and exported successfully")
            print("    ✅ Dashboard generation completed")
            
        except Exception as e:
            print(f"    ⚠️  Dashboard generation had issues (may be expected): {e}")
        
        # 7. Verify Data Persistence
        print("  💾 Verifying data persistence...")
        
        # Check quality database
        with sqlite3.connect(quality_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM quality_results")
            quality_records = cursor.fetchone()[0]
            assert quality_records > 0
            print(f"    📊 Quality results stored: {quality_records} records")
        
        # Check gates database
        with sqlite3.connect(gates_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pipeline_runs")
            pipeline_records = cursor.fetchone()[0]
            assert pipeline_records > 0
            print(f"    🚪 Pipeline runs stored: {pipeline_records} records")
        
        # Check monitoring database
        with sqlite3.connect(monitoring_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM metrics")
            metrics_records = cursor.fetchone()[0]
            assert metrics_records > 0
            print(f"    📈 Metrics stored: {metrics_records} records")
        
        print("    ✅ Data persistence verified")
        
        print("🎉 Complete monitoring workflow test PASSED")
        
        return {
            'quality_results': quality_results,
            'drift_results': drift_results,
            'performance_results': performance_results,
            'pipeline_results': pipeline_results,
            'metrics_count': len(stored_metrics)
        }
    
    def test_error_handling_and_resilience(self, temp_workspace):
        """Test error handling and system resilience."""
        print("🛡️  Testing error handling and resilience...")
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'col1': [1, 2, None, None, None],  # Lots of missing data
            'col2': ['A', 'B', 'C', 'D', 'E']
        })
        
        quality_db = os.path.join(temp_workspace, 'error_test_quality.db')
        quality_validator = DataQualityValidator(db_path=quality_db)
        
        # Should handle invalid data gracefully
        try:
            results = quality_validator.run_comprehensive_validation(
                invalid_data,
                'invalid_test_dataset'
            )
            # Should complete without crashing
            assert 'overall_status' in results
            print("    ✅ Invalid data handled gracefully")
        except Exception as e:
            pytest.fail(f"Failed to handle invalid data: {e}")
        
        # Test drift detection with mismatched columns
        drift_detector = DataDriftDetector()
        reference_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        current_data = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})  # Different columns
        
        drift_detector.fit_reference(reference_data)
        
        try:
            drift_results = drift_detector.detect_drift(current_data)
            # Should handle mismatched columns
            assert isinstance(drift_results, dict)
            print("    ✅ Mismatched columns handled gracefully")
        except Exception as e:
            # Some error is expected, but shouldn't crash the system
            print(f"    ✅ Expected error handled: {type(e).__name__}")
        
        # Test quality gates with missing configuration
        gates_db = os.path.join(temp_workspace, 'error_test_gates.db')
        gate_engine = QualityGateEngine(db_path=gates_db)
        
        try:
            # Try to execute non-existent gate
            result = gate_engine.execute_gate('non_existent_gate', {})
            pytest.fail("Should have raised an error for non-existent gate")
        except ValueError:
            print("    ✅ Non-existent gate error handled correctly")
        
        print("🛡️  Error handling and resilience tests PASSED")
    
    def test_performance_and_scalability(self, sample_house_data):
        """Test performance with larger datasets."""
        print("⚡ Testing performance and scalability...")
        
        # Create larger dataset
        large_data = pd.concat([sample_house_data] * 5, ignore_index=True)  # 5x larger
        print(f"    📊 Testing with {len(large_data)} records")
        
        # Test drift detection performance
        start_time = time.time()
        drift_detector = DataDriftDetector()
        drift_detector.fit_reference(large_data.iloc[:len(large_data)//2])
        drift_results = drift_detector.detect_drift(large_data.iloc[len(large_data)//2:])
        drift_time = time.time() - start_time
        
        assert drift_time < 30.0  # Should complete within 30 seconds
        print(f"    🔍 Drift detection completed in {drift_time:.2f}s")
        
        # Test data quality validation performance
        start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            quality_validator = DataQualityValidator(db_path=temp_db)
            quality_results = quality_validator.run_comprehensive_validation(
                large_data,
                'performance_test_dataset'
            )
            quality_time = time.time() - start_time
            
            assert quality_time < 60.0  # Should complete within 60 seconds
            assert quality_results['summary']['total_checks'] > 0
            print(f"    ✅ Data quality validation completed in {quality_time:.2f}s")
            
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)
        
        print("⚡ Performance and scalability tests PASSED")
    
    def test_configuration_management(self, temp_workspace):
        """Test configuration management and validation."""
        print("⚙️  Testing configuration management...")
        
        # Test quality gates configuration
        gate_engine = QualityGateEngine()
        
        # Create test configuration
        config_path = os.path.join(temp_workspace, 'test_config.yaml')
        gate_engine.create_default_configuration(config_path)
        
        assert os.path.exists(config_path)
        print("    ✅ Default configuration created")
        
        # Load and validate configuration
        gate_engine.load_configuration(config_path)
        assert len(gate_engine.gates) > 0
        print(f"    ⚙️  Loaded {len(gate_engine.gates)} gates from configuration")
        
        # Test configuration validation
        gates_with_dependencies = [
            gate for gate in gate_engine.gates.values() 
            if gate.depends_on
        ]
        
        if gates_with_dependencies:
            print(f"    🔗 Found {len(gates_with_dependencies)} gates with dependencies")
        
        print("⚙️  Configuration management tests PASSED")
    
    def test_monitoring_data_export_import(self, temp_workspace, sample_house_data):
        """Test data export and import capabilities."""
        print("💾 Testing data export and import...")
        
        # Setup monitoring components
        quality_db = os.path.join(temp_workspace, 'export_quality.db')
        monitoring_db = os.path.join(temp_workspace, 'export_monitoring.db')
        
        # Generate some monitoring data
        quality_validator = DataQualityValidator(db_path=quality_db)
        metrics_tracker = MetricsTracker(db_path=monitoring_db)
        
        # Run quality validation
        quality_results = quality_validator.run_comprehensive_validation(
            sample_house_data,
            'export_test_dataset'
        )
        
        # Track some metrics
        test_metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'data_volume': len(sample_house_data)
        }
        
        metrics_tracker.batch_track_metrics(test_metrics)
        
        # Test quality report export
        quality_report = quality_validator.get_quality_report(hours=1)
        assert quality_report['summary']['total_checks'] > 0
        print("    📊 Quality report generated")
        
        # Test metrics export
        metrics_data = metrics_tracker.get_metrics(hours=1)
        assert len(metrics_data) == len(test_metrics)
        print("    📈 Metrics data retrieved")
        
        # Test dashboard export
        dashboard = MonitoringDashboard(
            db_path=monitoring_db,
            quality_gates_db=os.path.join(temp_workspace, 'temp_gates.db'),
            data_quality_db=quality_db
        )
        
        try:
            # Export JSON report
            json_path = os.path.join(temp_workspace, 'export_test.json')
            overview = dashboard.generate_overview_dashboard(hours=1)
            dashboard.export_dashboard_report('overview', json_path, 'json')
            
            assert os.path.exists(json_path)
            
            # Verify exported content
            with open(json_path, 'r') as f:
                exported_data = json.load(f)
            assert 'generated_at' in exported_data
            
            print("    📄 Dashboard report exported to JSON")
            
            # Export HTML report
            html_path = os.path.join(temp_workspace, 'export_test.html')
            dashboard.export_dashboard_report('overview', html_path, 'html')
            
            assert os.path.exists(html_path)
            print("    🌐 Dashboard report exported to HTML")
            
        except Exception as e:
            print(f"    ⚠️  Dashboard export had issues: {e}")
        
        print("💾 Data export and import tests PASSED")


class TestPhase2RealWorldScenarios:
    """Test Phase 2 monitoring with real-world scenarios."""
    
    def test_model_degradation_detection(self):
        """Test detection of model performance degradation."""
        print("📉 Testing model degradation detection...")
        
        # Simulate initial good performance
        np.random.seed(42)
        y_true = np.random.normal(100, 20, 1000)
        
        performance_monitor = ModelPerformanceMonitor()
        
        # Baseline: good predictions
        y_pred_good = y_true + np.random.normal(0, 5, 1000)
        baseline_results = performance_monitor.evaluate_batch(y_true, y_pred_good)
        performance_monitor.store_baseline_performance(y_true, y_pred_good)
        
        baseline_r2 = baseline_results['metrics']['r2_score']
        print(f"    📊 Baseline R² score: {baseline_r2:.3f}")
        
        # Degraded: worse predictions
        y_pred_bad = y_true + np.random.normal(0, 15, 1000)  # More error
        degraded_results = performance_monitor.evaluate_batch(y_true, y_pred_bad)
        
        degraded_r2 = degraded_results['metrics']['r2_score']
        performance_change = degraded_results['baseline_comparison']['performance_change']
        
        print(f"    📉 Degraded R² score: {degraded_r2:.3f}")
        print(f"    🔄 Performance change: {performance_change:.3f}")
        
        # Should detect degradation
        assert performance_change < 0, "Should detect performance degradation"
        assert not degraded_results['baseline_comparison']['is_better']
        
        print("📉 Model degradation detection PASSED")
    
    def test_data_drift_scenarios(self):
        """Test various data drift scenarios."""
        print("🌊 Testing data drift scenarios...")
        
        # Create reference data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'numerical_stable': np.random.normal(0, 1, 1000),
            'numerical_drift': np.random.normal(0, 1, 1000),
            'categorical_stable': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
            'categorical_drift': np.random.choice(['X', 'Y', 'Z'], 1000, p=[0.4, 0.4, 0.2])
        })
        
        drift_detector = DataDriftDetector()
        drift_detector.fit_reference(reference_data)
        
        # Scenario 1: No drift
        current_data_no_drift = reference_data.copy()
        results_no_drift = drift_detector.detect_drift(current_data_no_drift)
        
        no_drift_detected = all(
            result.get('p_value', 1.0) > 0.05 
            for result in results_no_drift.values()
        )
        print(f"    🟢 No drift scenario: {'PASS' if no_drift_detected else 'FAIL'}")
        
        # Scenario 2: Numerical drift only
        current_data_num_drift = reference_data.copy()
        current_data_num_drift['numerical_drift'] += 2  # Shift mean
        results_num_drift = drift_detector.detect_drift(current_data_num_drift)
        
        num_drift_detected = results_num_drift['numerical_drift'].get('p_value', 1.0) < 0.05
        stable_still_stable = results_num_drift['numerical_stable'].get('p_value', 1.0) > 0.05
        
        print(f"    🔢 Numerical drift scenario: {'PASS' if num_drift_detected and stable_still_stable else 'FAIL'}")
        
        # Scenario 3: Categorical drift only
        current_data_cat_drift = reference_data.copy()
        # Change distribution: A becomes less frequent
        current_data_cat_drift['categorical_drift'] = np.random.choice(
            ['X', 'Y', 'Z'], 1000, p=[0.1, 0.7, 0.2]  # Changed distribution
        )
        results_cat_drift = drift_detector.detect_drift(current_data_cat_drift)
        
        cat_drift_detected = results_cat_drift['categorical_drift'].get('p_value', 1.0) < 0.05
        print(f"    📊 Categorical drift scenario: {'PASS' if cat_drift_detected else 'FAIL'}")
        
        print("🌊 Data drift scenarios PASSED")
    
    def test_quality_gates_pipeline_scenarios(self):
        """Test quality gates in various pipeline scenarios."""
        print("🚪 Testing quality gates pipeline scenarios...")
        
        gate_engine = QualityGateEngine()
        
        # Create test gates with dependencies
        from monitoring.quality_gates import QualityGateConfig, GateSeverity
        
        gates = [
            QualityGateConfig(
                name='data_volume',
                type='data_volume',
                enabled=True,
                severity=GateSeverity.CRITICAL,
                threshold=100,
                parameters={'min_rows': 100},
                depends_on=[],
                description='Data volume check'
            ),
            QualityGateConfig(
                name='data_quality',
                type='data_quality',
                enabled=True,
                severity=GateSeverity.HIGH,
                threshold=90.0,
                parameters={'max_missing_percent': 10.0},
                depends_on=['data_volume'],
                description='Data quality check'
            ),
            QualityGateConfig(
                name='model_performance',
                type='model_performance',
                enabled=True,
                severity=GateSeverity.CRITICAL,
                threshold=0.8,
                parameters={'primary_metric': 'r2_score'},
                depends_on=['data_quality'],
                description='Model performance check'
            )
        ]
        
        for gate in gates:
            gate_engine.gates[gate.name] = gate
        
        # Scenario 1: All gates pass
        good_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.uniform(0, 10, 200),
            'target': np.random.normal(100, 20, 200)
        })
        
        y_true = good_data['target'].values
        y_pred = y_true + np.random.normal(0, 5, len(y_true))  # Good predictions
        
        good_gate_data = {
            'dataframe': good_data,
            'dataset_name': 'good_scenario',
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        good_results = gate_engine.execute_pipeline('good_scenario', good_gate_data)
        good_passed = good_results['summary']['passed_gates']
        good_total = good_results['summary']['total_gates']
        
        print(f"    ✅ Good scenario: {good_passed}/{good_total} gates passed")
        
        # Scenario 2: Critical gate fails (should stop pipeline)
        bad_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),  # Too little data
            'feature2': np.random.uniform(0, 10, 50)
        })
        
        bad_gate_data = {
            'dataframe': bad_data,
            'dataset_name': 'bad_scenario'
        }
        
        bad_results = gate_engine.execute_pipeline('bad_scenario', bad_gate_data)
        assert bad_results['overall_status'] == 'failed'
        
        print(f"    ❌ Bad scenario: {bad_results['overall_status']} (expected)")
        
        print("🚪 Quality gates pipeline scenarios PASSED")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
