"""
Unit tests for data processor module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.processor import load_data, clean_data, process_data


class TestDataProcessor:
    """Test cases for data processor functions."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        self.sample_data = pd.DataFrame({
            'sqft': [1200, 1500, None, 2000, 800],
            'bedrooms': [2, 3, 2, 4, 1],
            'bathrooms': [1.5, 2.0, None, 2.5, 1.0],
            'location': ['urban', 'suburban', 'rural', None, 'urban'],
            'year_built': [1990, 2000, 1985, 2010, 1975],
            'condition': ['good', 'excellent', None, 'good', 'fair'],
            'price': [150000, 200000, 120000, 300000, 90000]
        })
        
        self.sample_data_with_outliers = pd.DataFrame({
            'sqft': [1200, 1500, 1800, 2000, 800],
            'bedrooms': [2, 3, 2, 4, 1],
            'bathrooms': [1.5, 2.0, 2.0, 2.5, 1.0],
            'location': ['urban', 'suburban', 'rural', 'urban', 'urban'],
            'year_built': [1990, 2000, 1985, 2010, 1975],
            'condition': ['good', 'excellent', 'good', 'good', 'fair'],
            'price': [150000, 200000, 120000, 1000000, 90000]  # 1M is outlier
        })
    
    @patch('data.processor.pd.read_csv')
    def test_load_data_success(self, mock_read_csv):
        """Test successful data loading."""
        # Arrange
        mock_read_csv.return_value = self.sample_data
        file_path = 'test_data.csv'
        
        # Act
        result = load_data(file_path)
        
        # Assert
        mock_read_csv.assert_called_once_with(file_path)
        pd.testing.assert_frame_equal(result, self.sample_data)
    
    @patch('data.processor.pd.read_csv')
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test data loading when file doesn't exist."""
        # Arrange
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent_file.csv')
    
    def test_clean_data_missing_values(self):
        """Test cleaning data with missing values."""
        # Act
        result = clean_data(self.sample_data)
        
        # Assert
        assert result.isnull().sum().sum() == 0, "Cleaned data should have no missing values"
        assert len(result) == len(self.sample_data), "Should preserve number of rows"
        
        # Check that numeric missing values are filled with median
        assert result.loc[2, 'sqft'] == self.sample_data['sqft'].median()
        assert result.loc[2, 'bathrooms'] == self.sample_data['bathrooms'].median()
        
        # Check that categorical missing values are filled with mode
        assert result.loc[3, 'location'] == self.sample_data['location'].mode()[0]
        assert result.loc[2, 'condition'] == self.sample_data['condition'].mode()[0]
    
    def test_clean_data_outliers_removal(self):
        """Test outlier removal in price column."""
        # Act
        result = clean_data(self.sample_data_with_outliers)
        
        # Assert
        # The outlier (1,000,000) should be removed
        assert len(result) == len(self.sample_data_with_outliers) - 1
        assert 1000000 not in result['price'].values
    
    def test_clean_data_no_missing_values(self):
        """Test cleaning data without missing values."""
        # Arrange
        clean_sample_data = self.sample_data.dropna()
        
        # Act
        result = clean_data(clean_sample_data)
        
        # Assert
        pd.testing.assert_frame_equal(result, clean_sample_data)
    
    def test_clean_data_empty_dataframe(self):
        """Test cleaning an empty dataframe."""
        # Arrange
        empty_df = pd.DataFrame()
        
        # Act
        result = clean_data(empty_df)
        
        # Assert
        assert len(result) == 0
        assert list(result.columns) == list(empty_df.columns)
    
    def test_clean_data_preserves_data_types(self):
        """Test that data cleaning preserves appropriate data types."""
        # Act
        result = clean_data(self.sample_data)
        
        # Assert
        assert pd.api.types.is_numeric_dtype(result['sqft'])
        assert pd.api.types.is_integer_dtype(result['bedrooms'])
        assert pd.api.types.is_numeric_dtype(result['bathrooms'])
        assert pd.api.types.is_integer_dtype(result['year_built'])
        assert pd.api.types.is_numeric_dtype(result['price'])
    
    def test_clean_data_with_all_missing_column(self):
        """Test cleaning data with a column that has all missing values."""
        # Arrange
        test_data = self.sample_data.copy()
        test_data['all_missing'] = None
        
        # Act
        result = clean_data(test_data)
        
        # Assert
        # Column with all missing values should be handled gracefully
        assert 'all_missing' in result.columns
    
    @patch('data.processor.clean_data')
    @patch('data.processor.load_data')
    def test_process_data_pipeline(self, mock_load_data, mock_clean_data):
        """Test the complete data processing pipeline."""
        # Arrange
        mock_load_data.return_value = self.sample_data
        mock_clean_data.return_value = self.sample_data.dropna()
        
        input_path = 'input.csv'
        output_path = 'output.csv'
        
        # Act
        with patch('data.processor.pd.DataFrame.to_csv') as mock_to_csv:
            from data.processor import process_data
            process_data(input_path, output_path)
        
        # Assert
        mock_load_data.assert_called_once_with(input_path)
        mock_clean_data.assert_called_once()
        mock_to_csv.assert_called_once_with(output_path, index=False)


class TestDataValidation:
    """Test cases for data validation."""
    
    def test_validate_data_schema(self):
        """Test data schema validation."""
        # This would be implemented with a schema validation library
        # like Great Expectations or Pandera
        pass
    
    def test_validate_data_ranges(self):
        """Test data range validation."""
        # Test that values are within expected ranges
        pass
    
    def test_validate_data_distributions(self):
        """Test data distribution validation."""
        # Test that data distributions are as expected
        pass


if __name__ == '__main__':
    pytest.main([__file__])
