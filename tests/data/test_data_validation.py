"""
Data validation tests using Great Expectations-style validation.
"""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import os
from datetime import datetime


class DataValidator:
    """Data validation class with Great Expectations-style validations."""
    
    def __init__(self):
        self.validation_results = []
    
    def expect_column_to_exist(self, df: pd.DataFrame, column: str) -> bool:
        """Validate that a column exists in the dataframe."""
        result = column in df.columns
        self.validation_results.append({
            "expectation": f"expect_column_to_exist_{column}",
            "success": result,
            "message": f"Column '{column}' {'exists' if result else 'does not exist'}"
        })
        return result
    
    def expect_column_values_to_be_of_type(self, df: pd.DataFrame, column: str, expected_type: str) -> bool:
        """Validate column data type."""
        if column not in df.columns:
            result = False
            message = f"Column '{column}' does not exist"
        else:
            if expected_type == "numeric":
                result = pd.api.types.is_numeric_dtype(df[column])
            elif expected_type == "string":
                result = pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column])
            elif expected_type == "integer":
                result = pd.api.types.is_integer_dtype(df[column])
            elif expected_type == "float":
                result = pd.api.types.is_float_dtype(df[column])
            else:
                result = False
            
            message = f"Column '{column}' {'is' if result else 'is not'} of type {expected_type}"
        
        self.validation_results.append({
            "expectation": f"expect_column_values_to_be_of_type_{column}_{expected_type}",
            "success": result,
            "message": message
        })
        return result
    
    def expect_column_values_to_not_be_null(self, df: pd.DataFrame, column: str) -> bool:
        """Validate that column has no null values."""
        if column not in df.columns:
            result = False
            message = f"Column '{column}' does not exist"
        else:
            null_count = df[column].isnull().sum()
            result = null_count == 0
            message = f"Column '{column}' has {null_count} null values"
        
        self.validation_results.append({
            "expectation": f"expect_column_values_to_not_be_null_{column}",
            "success": result,
            "message": message
        })
        return result
    
    def expect_column_values_to_be_between(self, df: pd.DataFrame, column: str, min_value: float, max_value: float) -> bool:
        """Validate that column values are within specified range."""
        if column not in df.columns:
            result = False
            message = f"Column '{column}' does not exist"
        else:
            out_of_range = ((df[column] < min_value) | (df[column] > max_value)).sum()
            result = out_of_range == 0
            message = f"Column '{column}' has {out_of_range} values outside range [{min_value}, {max_value}]"
        
        self.validation_results.append({
            "expectation": f"expect_column_values_to_be_between_{column}_{min_value}_{max_value}",
            "success": result,
            "message": message
        })
        return result
    
    def expect_column_values_to_be_in_set(self, df: pd.DataFrame, column: str, value_set: List[str]) -> bool:
        """Validate that column values are in specified set."""
        if column not in df.columns:
            result = False
            message = f"Column '{column}' does not exist"
        else:
            invalid_values = df[~df[column].isin(value_set)][column].unique()
            result = len(invalid_values) == 0
            message = f"Column '{column}' has invalid values: {list(invalid_values)}" if not result else f"All values in '{column}' are valid"
        
        self.validation_results.append({
            "expectation": f"expect_column_values_to_be_in_set_{column}",
            "success": result,
            "message": message
        })
        return result
    
    def expect_column_values_to_be_unique(self, df: pd.DataFrame, column: str) -> bool:
        """Validate that column values are unique."""
        if column not in df.columns:
            result = False
            message = f"Column '{column}' does not exist"
        else:
            duplicate_count = df[column].duplicated().sum()
            result = duplicate_count == 0
            message = f"Column '{column}' has {duplicate_count} duplicate values"
        
        self.validation_results.append({
            "expectation": f"expect_column_values_to_be_unique_{column}",
            "success": result,
            "message": message
        })
        return result
    
    def expect_table_row_count_to_be_between(self, df: pd.DataFrame, min_value: int, max_value: int) -> bool:
        """Validate that table has row count within specified range."""
        row_count = len(df)
        result = min_value <= row_count <= max_value
        message = f"Table has {row_count} rows, expected between {min_value} and {max_value}"
        
        self.validation_results.append({
            "expectation": f"expect_table_row_count_to_be_between_{min_value}_{max_value}",
            "success": result,
            "message": message
        })
        return result
    
    def expect_column_mean_to_be_between(self, df: pd.DataFrame, column: str, min_value: float, max_value: float) -> bool:
        """Validate that column mean is within specified range."""
        if column not in df.columns:
            result = False
            message = f"Column '{column}' does not exist"
        else:
            try:
                mean_value = df[column].mean()
                result = min_value <= mean_value <= max_value
                message = f"Column '{column}' mean is {mean_value:.2f}, expected between {min_value} and {max_value}"
            except TypeError:
                result = False
                message = f"Column '{column}' is not numeric, cannot calculate mean"
        
        self.validation_results.append({
            "expectation": f"expect_column_mean_to_be_between_{column}_{min_value}_{max_value}",
            "success": result,
            "message": message
        })
        return result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        total_expectations = len(self.validation_results)
        successful_expectations = sum(1 for result in self.validation_results if result["success"])
        
        return {
            "total_expectations": total_expectations,
            "successful_expectations": successful_expectations,
            "success_percentage": (successful_expectations / total_expectations * 100) if total_expectations > 0 else 0,
            "validation_results": self.validation_results,
            "timestamp": datetime.now().isoformat()
        }


class TestDataValidation:
    """Test cases for data validation."""
    
    def setup_method(self):
        """Set up test data."""
        self.validator = DataValidator()
        
        # Sample valid data
        self.valid_data = pd.DataFrame({
            'sqft': [1200, 1500, 1800, 2000, 800],
            'bedrooms': [2, 3, 2, 4, 1],
            'bathrooms': [1.5, 2.0, 2.0, 2.5, 1.0],
            'location': ['urban', 'suburban', 'rural', 'urban', 'urban'],
            'year_built': [1990, 2000, 1985, 2010, 1975],
            'condition': ['good', 'excellent', 'good', 'good', 'fair'],
            'price': [150000, 200000, 120000, 300000, 90000]
        })
        
        # Sample invalid data
        self.invalid_data = pd.DataFrame({
            'sqft': [1200, -100, 1800, None, 800],  # Negative and null values
            'bedrooms': [2, 3, 2, 4, 0],  # Zero bedrooms
            'bathrooms': [1.5, 2.0, None, 2.5, 1.0],  # Null value
            'location': ['urban', 'invalid_loc', 'rural', 'urban', 'urban'],  # Invalid location
            'year_built': [1990, 2000, 1700, 2010, 1975],  # Invalid year
            'condition': ['good', 'excellent', 'good', 'good', 'fair'],
            'price': [150000, 200000, 120000, -50000, 90000]  # Negative price
        })
    
    def test_validate_raw_house_data_schema(self):
        """Test validation of raw house data schema."""
        # Required columns
        required_columns = ['sqft', 'bedrooms', 'bathrooms', 'location', 'year_built', 'condition', 'price']
        
        # Validate all required columns exist
        for column in required_columns:
            assert self.validator.expect_column_to_exist(self.valid_data, column)
        
        # Validate data types
        assert self.validator.expect_column_values_to_be_of_type(self.valid_data, 'sqft', 'numeric')
        assert self.validator.expect_column_values_to_be_of_type(self.valid_data, 'bedrooms', 'integer')
        assert self.validator.expect_column_values_to_be_of_type(self.valid_data, 'bathrooms', 'numeric')
        assert self.validator.expect_column_values_to_be_of_type(self.valid_data, 'year_built', 'integer')
        assert self.validator.expect_column_values_to_be_of_type(self.valid_data, 'price', 'numeric')
    
    def test_validate_house_data_ranges(self):
        """Test validation of house data value ranges."""
        # Validate sqft range
        assert self.validator.expect_column_values_to_be_between(self.valid_data, 'sqft', 100, 50000)
        
        # Validate bedrooms range
        assert self.validator.expect_column_values_to_be_between(self.valid_data, 'bedrooms', 1, 20)
        
        # Validate bathrooms range
        assert self.validator.expect_column_values_to_be_between(self.valid_data, 'bathrooms', 0.5, 20)
        
        # Validate year_built range
        current_year = datetime.now().year
        assert self.validator.expect_column_values_to_be_between(self.valid_data, 'year_built', 1800, current_year)
        
        # Validate price range
        assert self.validator.expect_column_values_to_be_between(self.valid_data, 'price', 1000, 10000000)
    
    def test_validate_categorical_values(self):
        """Test validation of categorical column values."""
        # Validate location values
        valid_locations = ['urban', 'suburban', 'rural']
        assert self.validator.expect_column_values_to_be_in_set(self.valid_data, 'location', valid_locations)
        
        # Validate condition values
        valid_conditions = ['poor', 'fair', 'good', 'excellent']
        assert self.validator.expect_column_values_to_be_in_set(self.valid_data, 'condition', valid_conditions)
    
    def test_validate_no_null_values(self):
        """Test validation that required columns have no null values."""
        critical_columns = ['sqft', 'bedrooms', 'bathrooms', 'price']
        
        for column in critical_columns:
            assert self.validator.expect_column_values_to_not_be_null(self.valid_data, column)
    
    def test_validate_data_quality_metrics(self):
        """Test validation of data quality metrics."""
        # Validate row count
        assert self.validator.expect_table_row_count_to_be_between(self.valid_data, 1, 1000000)
        
        # Validate price mean (example range)
        assert self.validator.expect_column_mean_to_be_between(self.valid_data, 'price', 50000, 500000)
        
        # Validate sqft mean
        assert self.validator.expect_column_mean_to_be_between(self.valid_data, 'sqft', 800, 3000)
    
    def test_validation_with_invalid_data(self):
        """Test validation with invalid data to ensure proper failure detection."""
        # This should fail validations
        assert not self.validator.expect_column_values_to_be_between(self.invalid_data, 'sqft', 100, 50000)
        assert not self.validator.expect_column_values_to_be_between(self.invalid_data, 'bedrooms', 1, 20)
        assert not self.validator.expect_column_values_to_not_be_null(self.invalid_data, 'sqft')
        assert not self.validator.expect_column_values_to_be_in_set(self.invalid_data, 'location', ['urban', 'suburban', 'rural'])
    
    def test_validation_summary_generation(self):
        """Test generation of validation summary."""
        # Run some validations
        self.validator.expect_column_to_exist(self.valid_data, 'sqft')
        self.validator.expect_column_values_to_be_of_type(self.valid_data, 'sqft', 'numeric')
        self.validator.expect_column_values_to_not_be_null(self.valid_data, 'sqft')
        
        # Get summary
        summary = self.validator.get_validation_summary()
        
        # Assert
        assert "total_expectations" in summary
        assert "successful_expectations" in summary
        assert "success_percentage" in summary
        assert "validation_results" in summary
        assert "timestamp" in summary
        assert summary["total_expectations"] == 3
        assert summary["successful_expectations"] == 3
        assert summary["success_percentage"] == 100.0
    
    def test_validate_processed_data(self):
        """Test validation of processed data after cleaning."""
        # Simulate processed data (no nulls, valid ranges)
        processed_data = self.valid_data.copy()
        
        # All critical validations should pass for processed data
        critical_validations = [
            self.validator.expect_column_values_to_not_be_null(processed_data, 'sqft'),
            self.validator.expect_column_values_to_not_be_null(processed_data, 'price'),
            self.validator.expect_column_values_to_be_between(processed_data, 'price', 0, 10000000),
            self.validator.expect_column_values_to_be_between(processed_data, 'sqft', 1, 50000)
        ]
        
        assert all(critical_validations)
    
    def test_validate_feature_engineered_data(self):
        """Test validation of feature-engineered data."""
        # Simulate feature-engineered data
        featured_data = self.valid_data.copy()
        featured_data['house_age'] = 2024 - featured_data['year_built']
        featured_data['bed_bath_ratio'] = featured_data['bedrooms'] / featured_data['bathrooms']
        featured_data['price_per_sqft'] = featured_data['price'] / featured_data['sqft']
        
        # Validate new features exist
        assert self.validator.expect_column_to_exist(featured_data, 'house_age')
        assert self.validator.expect_column_to_exist(featured_data, 'bed_bath_ratio')
        assert self.validator.expect_column_to_exist(featured_data, 'price_per_sqft')
        
        # Validate new feature ranges
        assert self.validator.expect_column_values_to_be_between(featured_data, 'house_age', 0, 300)
        assert self.validator.expect_column_values_to_be_between(featured_data, 'bed_bath_ratio', 0.1, 20)
        assert self.validator.expect_column_values_to_be_between(featured_data, 'price_per_sqft', 1, 1000)


def validate_data_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a data file and return validation results.
    
    Args:
        file_path: Path to the CSV data file
    
    Returns:
        Dictionary containing validation results
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Create validator
        validator = DataValidator()
        
        # Run standard validations
        required_columns = ['sqft', 'bedrooms', 'bathrooms', 'location', 'year_built', 'condition', 'price']
        
        # Schema validation
        for column in required_columns:
            validator.expect_column_to_exist(df, column)
        
        # Data type validation
        validator.expect_column_values_to_be_of_type(df, 'sqft', 'numeric')
        validator.expect_column_values_to_be_of_type(df, 'bedrooms', 'integer')
        validator.expect_column_values_to_be_of_type(df, 'bathrooms', 'numeric')
        validator.expect_column_values_to_be_of_type(df, 'year_built', 'integer')
        validator.expect_column_values_to_be_of_type(df, 'price', 'numeric')
        
        # Range validation
        validator.expect_column_values_to_be_between(df, 'sqft', 100, 50000)
        validator.expect_column_values_to_be_between(df, 'bedrooms', 1, 20)
        validator.expect_column_values_to_be_between(df, 'bathrooms', 0.5, 20)
        validator.expect_column_values_to_be_between(df, 'year_built', 1800, datetime.now().year)
        validator.expect_column_values_to_be_between(df, 'price', 1000, 10000000)
        
        # Categorical validation
        validator.expect_column_values_to_be_in_set(df, 'location', ['urban', 'suburban', 'rural'])
        validator.expect_column_values_to_be_in_set(df, 'condition', ['poor', 'fair', 'good', 'excellent'])
        
        # Null value validation
        for column in ['sqft', 'bedrooms', 'bathrooms', 'price']:
            validator.expect_column_values_to_not_be_null(df, column)
        
        # Row count validation
        validator.expect_table_row_count_to_be_between(df, 1, 1000000)
        
        return validator.get_validation_summary()
        
    except Exception as e:
        return {
            "error": str(e),
            "success_percentage": 0,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == '__main__':
    pytest.main([__file__])
