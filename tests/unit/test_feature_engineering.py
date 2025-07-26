"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from features.engineer import create_features


class TestFeatureEngineering:
    """Test cases for feature engineering functions."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        self.sample_data = pd.DataFrame({
            'sqft': [1200, 1500, 1800, 2000, 800],
            'bedrooms': [2, 3, 2, 4, 1],
            'bathrooms': [1.5, 2.0, 2.0, 2.5, 1.0],
            'location': ['urban', 'suburban', 'rural', 'urban', 'urban'],
            'year_built': [1990, 2000, 1985, 2010, 1975],
            'condition': ['good', 'excellent', 'good', 'good', 'fair'],
            'price': [150000, 200000, 120000, 300000, 90000]
        })
    
    def test_create_features_adds_house_age(self):
        """Test that house age feature is created correctly."""
        # Act
        result = create_features(self.sample_data)
        
        # Assert
        assert 'house_age' in result.columns
        current_year = datetime.now().year
        expected_ages = current_year - self.sample_data['year_built']
        pd.testing.assert_series_equal(
            result['house_age'], 
            expected_ages,
            check_names=False
        )
    
    def test_create_features_adds_price_per_sqft(self):
        """Test that price per sqft feature is created correctly."""
        # Act
        result = create_features(self.sample_data)
        
        # Assert
        assert 'price_per_sqft' in result.columns
        expected_price_per_sqft = self.sample_data['price'] / self.sample_data['sqft']
        pd.testing.assert_series_equal(
            result['price_per_sqft'],
            expected_price_per_sqft,
            check_names=False
        )
    
    def test_create_features_adds_bed_bath_ratio(self):
        """Test that bed to bath ratio feature is created correctly."""
        # Act
        result = create_features(self.sample_data)
        
        # Assert
        assert 'bed_bath_ratio' in result.columns
        expected_ratios = self.sample_data['bedrooms'] / self.sample_data['bathrooms']
        pd.testing.assert_series_equal(
            result['bed_bath_ratio'],
            expected_ratios,
            check_names=False
        )
    
    def test_create_features_handles_division_by_zero(self):
        """Test that division by zero in bed_bath_ratio is handled."""
        # Arrange
        test_data = pd.DataFrame({
            'sqft': [1200],
            'bedrooms': [2],
            'bathrooms': [0],  # Zero bathrooms
            'location': ['urban'],
            'year_built': [2000],
            'condition': ['good'],
            'price': [150000]
        })
        
        # Act
        result = create_features(test_data)
        
        # Assert
        # Should handle division by zero gracefully (should be 0 after fillna)
        assert result['bed_bath_ratio'].iloc[0] == 0
    
    def test_create_features_preserves_original_columns(self):
        """Test that original columns are preserved."""
        # Act
        result = create_features(self.sample_data)
        
        # Assert
        original_columns = set(self.sample_data.columns)
        result_columns = set(result.columns)
        assert original_columns.issubset(result_columns)
    
    def test_create_features_preserves_row_count(self):
        """Test that the number of rows is preserved."""
        # Act
        result = create_features(self.sample_data)
        
        # Assert
        assert len(result) == len(self.sample_data)
    
    def test_create_features_with_empty_dataframe(self):
        """Test feature creation with empty dataframe."""
        # Arrange
        empty_df = pd.DataFrame()
        
        # Act & Assert
        # This might raise an error, which is expected behavior
        with pytest.raises(KeyError):
            create_features(empty_df)
    
    def test_create_features_with_missing_columns(self):
        """Test feature creation with missing required columns."""
        # Arrange
        incomplete_data = pd.DataFrame({
            'sqft': [1200, 1500],
            'bedrooms': [2, 3]
            # Missing other required columns
        })
        
        # Act & Assert
        with pytest.raises(KeyError):
            create_features(incomplete_data)
    
    def test_create_features_data_types(self):
        """Test that created features have correct data types."""
        # Act
        result = create_features(self.sample_data)
        
        # Assert
        assert pd.api.types.is_numeric_dtype(result['house_age'])
        assert pd.api.types.is_numeric_dtype(result['bed_bath_ratio'])
        assert pd.api.types.is_numeric_dtype(result['price_per_sqft'])
    
    def test_create_features_does_not_modify_original(self):
        """Test that the original dataframe is not modified."""
        # Arrange
        original_columns = set(self.sample_data.columns)
        
        # Act
        create_features(self.sample_data)
        
        # Assert
        assert set(self.sample_data.columns) == original_columns


class TestFeatureValidation:
    """Test cases for feature validation."""
    
    def test_validate_feature_ranges(self):
        """Test that created features are within expected ranges."""
        # Arrange
        sample_data = pd.DataFrame({
            'sqft': [1200, 1500, 1800],
            'bedrooms': [2, 3, 2],
            'bathrooms': [1.5, 2.0, 2.0],
            'location': ['urban', 'suburban', 'rural'],
            'year_built': [1990, 2000, 1985],
            'condition': ['good', 'excellent', 'good'],
            'price': [150000, 200000, 120000]
        })
        
        # Act
        result = create_features(sample_data)
        
        # Assert
        # House age should be non-negative
        assert (result['house_age'] >= 0).all()
        
        # Bed-bath ratio should be non-negative (after handling inf values)
        assert (result['bed_bath_ratio'] >= 0).all()
        
        # Price per sqft should be positive
        assert (result['price_per_sqft'] > 0).all()
    
    def test_validate_feature_consistency(self):
        """Test that created features are consistent with input data."""
        # Arrange
        sample_data = pd.DataFrame({
            'sqft': [1200],
            'bedrooms': [2],
            'bathrooms': [1.5],
            'location': ['urban'],
            'year_built': [2000],
            'condition': ['good'],
            'price': [150000]
        })
        
        # Act
        result = create_features(sample_data)
        
        # Assert
        current_year = datetime.now().year
        expected_age = current_year - 2000
        assert result['house_age'].iloc[0] == expected_age
        
        expected_price_per_sqft = 150000 / 1200
        assert result['price_per_sqft'].iloc[0] == expected_price_per_sqft
        
        expected_bed_bath_ratio = 2 / 1.5
        assert abs(result['bed_bath_ratio'].iloc[0] - expected_bed_bath_ratio) < 0.001


if __name__ == '__main__':
    pytest.main([__file__])
