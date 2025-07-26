"""
Unit tests for utils module.
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path
import tempfile

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'api'))

from utils import (
    validate_model_files, 
    validate_input_data, 
    sanitize_prediction_output,
    health_check,
    get_model_info,
    setup_logging
)


class TestUtilsFunctions:
    """Test cases for utility functions."""
    
    def test_validate_input_data_valid(self):
        """Test input validation with valid data."""
        # Arrange
        valid_data = {
            "sqft": 1500.0,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "suburban",
            "year_built": 2000,
            "condition": "good"
        }
        
        # Act
        result = validate_input_data(valid_data)
        
        # Assert
        assert result["sqft"] == 1500.0
        assert result["bedrooms"] == 3
        assert result["bathrooms"] == 2.0
        assert result["location"] == "suburban"
        assert result["year_built"] == 2000
        assert result["condition"] == "good"
    
    def test_validate_input_data_missing_field(self):
        """Test input validation with missing required field."""
        # Arrange
        invalid_data = {
            "sqft": 1500.0,
            "bedrooms": 3
            # Missing other required fields
        }
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_input_data(invalid_data)
        assert "Missing required field" in str(exc_info.value)
    
    def test_validate_input_data_invalid_sqft(self):
        """Test input validation with invalid sqft."""
        # Arrange
        invalid_data = {
            "sqft": -100,  # Invalid negative value
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "suburban",
            "year_built": 2000,
            "condition": "good"
        }
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_input_data(invalid_data)
        assert "sqft must be positive" in str(exc_info.value)
    
    def test_validate_input_data_invalid_bedrooms(self):
        """Test input validation with invalid bedrooms."""
        # Arrange
        invalid_data = {
            "sqft": 1500.0,
            "bedrooms": 0,  # Invalid: must be >= 1
            "bathrooms": 2.0,
            "location": "suburban",
            "year_built": 2000,
            "condition": "good"
        }
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_input_data(invalid_data)
        assert "bedrooms must be between 1 and 20" in str(exc_info.value)
    
    def test_validate_input_data_invalid_year(self):
        """Test input validation with invalid year."""
        # Arrange
        invalid_data = {
            "sqft": 1500.0,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "suburban",
            "year_built": 1700,  # Invalid: before 1800
            "condition": "good"
        }
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_input_data(invalid_data)
        assert "year_built must be between 1800" in str(exc_info.value)
    
    def test_validate_input_data_invalid_location(self):
        """Test input validation with invalid location."""
        # Arrange
        invalid_data = {
            "sqft": 1500.0,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "invalid_location",
            "year_built": 2000,
            "condition": "good"
        }
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_input_data(invalid_data)
        assert "location must be one of" in str(exc_info.value)
    
    def test_validate_input_data_case_insensitive(self):
        """Test that location and condition validation is case insensitive."""
        # Arrange
        valid_data = {
            "sqft": 1500.0,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "SUBURBAN",  # Uppercase
            "year_built": 2000,
            "condition": "GOOD"  # Uppercase
        }
        
        # Act
        result = validate_input_data(valid_data)
        
        # Assert
        assert result["location"] == "suburban"  # Should be lowercase
        assert result["condition"] == "good"  # Should be lowercase
    
    def test_validate_model_files_success(self):
        """Test model file validation with existing files."""
        # Arrange
        with tempfile.NamedTemporaryFile(delete=False) as model_file:
            model_file.write(b"dummy model content")
            model_path = model_file.name
        
        with tempfile.NamedTemporaryFile(delete=False) as preprocessor_file:
            preprocessor_file.write(b"dummy preprocessor content")
            preprocessor_path = preprocessor_file.name
        
        try:
            # Act
            result = validate_model_files(model_path, preprocessor_path)
            
            # Assert
            assert result is True
        finally:
            # Cleanup
            os.unlink(model_path)
            os.unlink(preprocessor_path)
    
    def test_validate_model_files_missing_model(self):
        """Test model file validation with missing model file."""
        # Arrange
        missing_model_path = "nonexistent_model.pkl"
        
        with tempfile.NamedTemporaryFile(delete=False) as preprocessor_file:
            preprocessor_file.write(b"dummy preprocessor content")
            preprocessor_path = preprocessor_file.name
        
        try:
            # Act
            result = validate_model_files(missing_model_path, preprocessor_path)
            
            # Assert
            assert result is False
        finally:
            # Cleanup
            os.unlink(preprocessor_path)
    
    def test_validate_model_files_empty_file(self):
        """Test model file validation with empty file."""
        # Arrange
        with tempfile.NamedTemporaryFile(delete=False) as model_file:
            # Don't write anything - file will be empty
            model_path = model_file.name
        
        with tempfile.NamedTemporaryFile(delete=False) as preprocessor_file:
            preprocessor_file.write(b"dummy preprocessor content")
            preprocessor_path = preprocessor_file.name
        
        try:
            # Act
            result = validate_model_files(model_path, preprocessor_path)
            
            # Assert
            assert result is False
        finally:
            # Cleanup
            os.unlink(model_path)
            os.unlink(preprocessor_path)
    
    def test_sanitize_prediction_output_valid(self):
        """Test prediction output sanitization with valid data."""
        # Arrange
        prediction = 200000.50
        confidence_interval = [180000.25, 220000.75]
        
        # Act
        result = sanitize_prediction_output(prediction, confidence_interval)
        
        # Assert
        assert result["predicted_price"] == 200000.50
        assert result["confidence_interval"] == [180000.25, 220000.75]
        assert "timestamp" in result
    
    def test_sanitize_prediction_output_negative_prediction(self):
        """Test prediction output sanitization with negative prediction."""
        # Arrange
        prediction = -50000
        confidence_interval = [-60000, -40000]
        
        # Act
        result = sanitize_prediction_output(prediction, confidence_interval)
        
        # Assert
        assert result["predicted_price"] == 0.0  # Should be clamped to 0
    
    def test_sanitize_prediction_output_numpy_types(self):
        """Test prediction output sanitization with numpy types."""
        # Arrange
        import numpy as np
        prediction = np.float64(200000.123456)
        confidence_interval = [np.float64(180000.123456), np.float64(220000.123456)]
        
        # Act
        result = sanitize_prediction_output(prediction, confidence_interval)
        
        # Assert
        assert isinstance(result["predicted_price"], float)
        assert all(isinstance(ci, float) for ci in result["confidence_interval"])
        assert result["predicted_price"] == 200000.12  # Rounded to 2 decimals
    
    def test_get_model_info(self):
        """Test getting model information."""
        # Act
        result = get_model_info()
        
        # Assert
        assert "model_version" in result
        assert "model_type" in result
        assert "features" in result
        assert "target" in result
        assert "performance_metrics" in result
        assert isinstance(result["features"], list)
        assert isinstance(result["performance_metrics"], dict)
    
    @patch('utils.validate_model_files')
    @patch('utils.psutil.Process')
    @patch('utils.psutil.disk_usage')
    def test_health_check_healthy(self, mock_disk_usage, mock_process, mock_validate_files):
        """Test health check when system is healthy."""
        # Arrange
        mock_validate_files.return_value = True
        
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        mock_disk_info = MagicMock()
        mock_disk_info.total = 1000 * 1024 * 1024 * 1024  # 1 TB
        mock_disk_info.used = 500 * 1024 * 1024 * 1024    # 500 GB
        mock_disk_usage.return_value = mock_disk_info
        
        # Act
        result = health_check()
        
        # Assert
        assert result["status"] == "healthy"
        assert result["checks"]["model_files"] is True
        assert "memory_usage_mb" in result["checks"]
        assert "disk_usage_percent" in result["checks"]
    
    @patch('utils.validate_model_files')
    def test_health_check_unhealthy_model_files(self, mock_validate_files):
        """Test health check when model files are invalid."""
        # Arrange
        mock_validate_files.return_value = False
        
        # Act
        result = health_check()
        
        # Assert
        assert result["status"] == "unhealthy"
        assert result["checks"]["model_files"] is False
    
    def test_setup_logging(self):
        """Test logging setup."""
        # Act
        logger = setup_logging(log_level="DEBUG")
        
        # Assert
        assert logger.level <= 10  # DEBUG level is 10
        assert len(logger.handlers) > 0


class TestDecorators:
    """Test cases for decorator functions."""
    
    @patch('utils.logger')
    def test_log_execution_time_decorator(self, mock_logger):
        """Test execution time logging decorator."""
        # This would require importing and testing the decorators
        # For now, we'll test the concept
        pass
    
    @patch('utils.logger')
    def test_handle_errors_decorator(self, mock_logger):
        """Test error handling decorator."""
        # This would require importing and testing the decorators
        # For now, we'll test the concept
        pass


if __name__ == '__main__':
    pytest.main([__file__])
