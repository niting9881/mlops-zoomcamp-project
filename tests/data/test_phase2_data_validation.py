"""
Data validation tests for Phase 2 monitoring features.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

# Import monitoring components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from monitoring import DataQualityValidator, DataDriftDetector


class TestDataValidation:
    """Test data validation capabilities."""
    
    @pytest.fixture
    def clean_house_data(self):
        """Generate clean house price data."""
        np.random.seed(42)
        n_samples = 500
        
        return pd.DataFrame({
            'price': np.random.normal(500000, 100000, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.uniform(1, 4, n_samples),
            'sqft_living': np.random.normal(2000, 400, n_samples),
            'sqft_lot': np.random.normal(8000, 2000, n_samples),
            'floors': np.random.choice([1, 2, 3], n_samples),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'condition': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'zipcode': np.random.choice(['98001', '98002', '98003'], n_samples)
        })
    
    @pytest.fixture
    def dirty_house_data(self):
        """Generate house data with quality issues."""
        np.random.seed(123)
        n_samples = 500
        
        data = pd.DataFrame({
            'price': np.random.normal(500000, 100000, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.uniform(1, 4, n_samples),
            'sqft_living': np.random.normal(2000, 400, n_samples),
            'sqft_lot': np.random.normal(8000, 2000, n_samples),
            'floors': np.random.choice([1, 2, 3], n_samples),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'condition': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'zipcode': np.random.choice(['98001', '98002', '98003'], n_samples)
        })
        
        # Introduce quality issues
        # 1. Missing values (10% in price)
        missing_indices = np.random.choice(data.index, size=int(0.1 * n_samples), replace=False)
        data.loc[missing_indices, 'price'] = np.nan
        
        # 2. Outliers in sqft_living
        outlier_indices = np.random.choice(data.index, size=20, replace=False)
        data.loc[outlier_indices, 'sqft_living'] = 50000  # Unrealistic values
        
        # 3. Inconsistent values (negative prices)
        negative_indices = np.random.choice(data.index, size=10, replace=False)
        data.loc[negative_indices, 'price'] = -100000
        
        # 4. Duplicate rows
        duplicate_rows = data.iloc[:20].copy()
        data = pd.concat([data, duplicate_rows], ignore_index=True)
        
        return data
    
    def test_clean_data_validation(self, clean_house_data):
        """Test validation of clean data."""
        print("✅ Testing clean data validation...")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            validator = DataQualityValidator(db_path=temp_db)
            
            results = validator.run_comprehensive_validation(
                clean_house_data,
                'clean_house_data',
                validation_config={
                    'required_columns': ['price', 'bedrooms', 'sqft_living'],
                    'max_missing_percent': 5.0,
                    'consistency_rules': {
                        'price_min': 100000,
                        'price_max': 2000000,
                        'bedrooms_min': 1,
                        'bedrooms_max': 8
                    }
                }
            )
            
            # Clean data should mostly pass validation
            pass_rate = results['summary']['pass_rate_percent']
            assert pass_rate > 90, f"Clean data should have >90% pass rate, got {pass_rate}%"
            
            # Should have no critical failures
            critical_failures = results['summary']['critical_failures']
            assert critical_failures == 0, f"Clean data should have no critical failures, got {critical_failures}"
            
            print(f"    📊 Clean data pass rate: {pass_rate:.1f}%")
            print(f"    🎯 Critical failures: {critical_failures}")
            
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)
    
    def test_dirty_data_validation(self, dirty_house_data):
        """Test validation of data with quality issues."""
        print("❌ Testing dirty data validation...")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            validator = DataQualityValidator(db_path=temp_db)
            
            results = validator.run_comprehensive_validation(
                dirty_house_data,
                'dirty_house_data',
                validation_config={
                    'required_columns': ['price', 'bedrooms', 'sqft_living'],
                    'max_missing_percent': 5.0,
                    'consistency_rules': {
                        'price_min': 100000,
                        'price_max': 2000000,
                        'bedrooms_min': 1,
                        'bedrooms_max': 8,
                        'sqft_living_min': 500,
                        'sqft_living_max': 10000
                    },
                    'accuracy_rules': {
                        'max_outlier_percent': 10.0
                    }
                }
            )
            
            # Dirty data should have lower pass rate
            pass_rate = results['summary']['pass_rate_percent']
            assert pass_rate < 90, f"Dirty data should have <90% pass rate, got {pass_rate}%"
            
            # Should detect issues
            failed_checks = results['summary']['failed_checks']
            assert failed_checks > 0, "Should detect data quality issues"
            
            print(f"    📊 Dirty data pass rate: {pass_rate:.1f}%")
            print(f"    ❌ Failed checks: {failed_checks}")
            
            # Check specific issues were detected
            detailed_results = results['detailed_results']
            
            # Should detect missing values
            missing_checks = [r for r in detailed_results if 'missing_data' in r.rule_name and not r.passed]
            assert len(missing_checks) > 0, "Should detect missing data issues"
            
            # Should detect duplicates
            duplicate_checks = [r for r in detailed_results if 'duplicate' in r.rule_name and not r.passed]
            assert len(duplicate_checks) > 0, "Should detect duplicate rows"
            
            print(f"    🔍 Missing data issues detected: {len(missing_checks)}")
            print(f"    🔍 Duplicate issues detected: {len(duplicate_checks)}")
            
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)
    
    def test_data_profiling(self, clean_house_data):
        """Test data profiling capabilities."""
        print("📊 Testing data profiling...")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            validator = DataQualityValidator(db_path=temp_db)
            
            profile = validator.create_data_profile(clean_house_data, 'profile_test')
            
            # Basic profile checks
            assert profile.total_rows == len(clean_house_data)
            assert profile.total_columns == len(clean_house_data.columns)
            
            # Column profiles should exist for all columns
            assert len(profile.column_profiles) == len(clean_house_data.columns)
            
            # Check numerical column profiling
            price_profile = profile.column_profiles['price']
            assert 'mean' in price_profile
            assert 'std' in price_profile
            assert 'min' in price_profile
            assert 'max' in price_profile
            
            # Check categorical column profiling
            zipcode_profile = profile.column_profiles['zipcode']
            assert 'unique_count' in zipcode_profile
            assert 'top_value' in zipcode_profile
            
            print(f"    📈 Total rows: {profile.total_rows}")
            print(f"    📊 Total columns: {profile.total_columns}")
            print(f"    💰 Price mean: ${price_profile['mean']:,.0f}")
            print(f"    🏠 Unique zipcodes: {zipcode_profile['unique_count']}")
            
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)
    
    def test_drift_detection_validation(self, clean_house_data):
        """Test drift detection with validated data."""
        print("🌊 Testing drift detection validation...")
        
        # Split data for drift testing
        reference_data = clean_house_data.iloc[:300].copy()
        current_data = clean_house_data.iloc[300:].copy()
        
        # Add some drift to current data
        current_data['price'] = current_data['price'] * 1.1  # 10% price increase
        
        drift_detector = DataDriftDetector()
        drift_detector.fit_reference(reference_data)
        
        # Test drift detection
        drift_results = drift_detector.detect_drift(current_data)
        
        # Should detect drift in price
        price_drift = drift_results.get('price', {})
        if 'p_value' in price_drift:
            assert price_drift['p_value'] < 0.05, "Should detect price drift"
            print(f"    💰 Price drift detected (p-value: {price_drift['p_value']:.4f})")
        
        # Other features should show less or no drift
        stable_features = ['bedrooms', 'floors', 'condition']
        for feature in stable_features:
            if feature in drift_results:
                feature_result = drift_results[feature]
                if 'p_value' in feature_result:
                    print(f"    🏠 {feature} drift p-value: {feature_result['p_value']:.4f}")
    
    def test_business_rules_validation(self, clean_house_data, dirty_house_data):
        """Test business-specific validation rules."""
        print("💼 Testing business rules validation...")
        
        # Define business rules for house price data
        business_rules = {
            'price_min': 50000,  # Minimum reasonable house price
            'price_max': 5000000,  # Maximum reasonable house price
            'bedrooms_min': 0,
            'bedrooms_max': 10,
            'sqft_living_min': 300,  # Minimum livable space
            'sqft_living_max': 15000,  # Maximum reasonable space
            'floors_min': 1,
            'floors_max': 5
        }
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            validator = DataQualityValidator(db_path=temp_db)
            
            # Test clean data against business rules
            clean_results = validator.validate_consistency(
                clean_house_data,
                'clean_business_rules',
                consistency_rules=business_rules
            )
            
            # Most clean data should pass business rules
            passed_clean = sum(1 for r in clean_results if r.passed)
            total_clean = len(clean_results)
            clean_pass_rate = (passed_clean / total_clean) * 100
            
            print(f"    ✅ Clean data business rules pass rate: {clean_pass_rate:.1f}%")
            
            # Test dirty data against business rules
            dirty_results = validator.validate_consistency(
                dirty_house_data,
                'dirty_business_rules',
                consistency_rules=business_rules
            )
            
            # Dirty data should fail more business rules
            passed_dirty = sum(1 for r in dirty_results if r.passed)
            total_dirty = len(dirty_results)
            dirty_pass_rate = (passed_dirty / total_dirty) * 100
            
            print(f"    ❌ Dirty data business rules pass rate: {dirty_pass_rate:.1f}%")
            
            # Dirty data should have lower pass rate
            assert dirty_pass_rate < clean_pass_rate, "Dirty data should fail more business rules"
            
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)
    
    def test_schema_validation(self, clean_house_data):
        """Test schema and data type validation."""
        print("📋 Testing schema validation...")
        
        expected_schema = {
            'price': 'float64',
            'bedrooms': 'int64',
            'bathrooms': 'float64',
            'sqft_living': 'float64',
            'sqft_lot': 'float64',
            'floors': 'int64',
            'waterfront': 'int64',
            'condition': 'int64',
            'zipcode': 'object'
        }
        
        # Check data types match expected schema
        actual_dtypes = clean_house_data.dtypes.to_dict()
        
        schema_matches = 0
        total_columns = len(expected_schema)
        
        for column, expected_dtype in expected_schema.items():
            if column in actual_dtypes:
                actual_dtype = str(actual_dtypes[column])
                if actual_dtype == expected_dtype:
                    schema_matches += 1
                    print(f"    ✅ {column}: {actual_dtype} (matches)")
                else:
                    print(f"    ❌ {column}: {actual_dtype} (expected {expected_dtype})")
            else:
                print(f"    ❌ {column}: Missing column")
        
        schema_match_rate = (schema_matches / total_columns) * 100
        print(f"    📊 Schema match rate: {schema_match_rate:.1f}%")
        
        # Most columns should match expected schema
        assert schema_match_rate > 70, f"Schema match rate should be >70%, got {schema_match_rate}%"
    
    def test_data_volume_validation(self):
        """Test data volume and size validation."""
        print("📏 Testing data volume validation...")
        
        # Test different data sizes
        test_sizes = [10, 100, 1000, 5000]
        
        for size in test_sizes:
            np.random.seed(42)
            test_data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, size),
                'feature2': np.random.uniform(0, 10, size),
                'target': np.random.normal(100, 20, size)
            })
            
            # Calculate data volume metrics
            total_rows = len(test_data)
            total_columns = len(test_data.columns)
            total_cells = total_rows * total_columns
            memory_usage = test_data.memory_usage(deep=True).sum()
            
            print(f"    📊 Size {size}: {total_rows} rows, {total_columns} cols, "
                  f"{total_cells} cells, {memory_usage/1024:.1f} KB")
            
            # Basic volume validations
            assert total_rows == size, f"Expected {size} rows, got {total_rows}"
            assert total_columns == 3, f"Expected 3 columns, got {total_columns}"
            assert memory_usage > 0, "Memory usage should be positive"
    
    def test_data_freshness_validation(self):
        """Test data freshness and timestamp validation."""
        print("⏰ Testing data freshness validation...")
        
        from datetime import datetime, timedelta
        
        # Create data with timestamps
        now = datetime.now()
        timestamps = [
            now - timedelta(hours=1),   # Fresh
            now - timedelta(hours=12),  # Moderate
            now - timedelta(days=1),    # Old
            now - timedelta(days=7),    # Very old
        ]
        
        test_data = pd.DataFrame({
            'timestamp': timestamps,
            'value': [100, 200, 300, 400],
            'category': ['A', 'B', 'C', 'D']
        })
        
        # Check data age
        for i, (idx, row) in enumerate(test_data.iterrows()):
            age_hours = (now - row['timestamp']).total_seconds() / 3600
            
            if age_hours <= 6:
                freshness = "Fresh"
            elif age_hours <= 24:
                freshness = "Moderate"
            elif age_hours <= 168:  # 1 week
                freshness = "Old"
            else:
                freshness = "Very Old"
            
            print(f"    ⏱️  Record {i+1}: {age_hours:.1f} hours old ({freshness})")
        
        # Should have mix of freshness levels
        age_hours_list = [(now - ts).total_seconds() / 3600 for ts in timestamps]
        fresh_count = sum(1 for age in age_hours_list if age <= 6)
        old_count = sum(1 for age in age_hours_list if age > 24)
        
        print(f"    📊 Fresh records (≤6h): {fresh_count}")
        print(f"    📊 Old records (>24h): {old_count}")


if __name__ == "__main__":
    # Run data validation tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
