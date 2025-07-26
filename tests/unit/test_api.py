"""
Unit tests for the FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'api'))

from main import app
from schemas import HousePredictionRequest, PredictionResponse


class TestAPI:
    """Test cases for FastAPI endpoints."""
    
    def setup_method(self):
        """Set up test client and sample data."""
        self.client = TestClient(app)
        self.sample_request = {
            "sqft": 1500.0,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "suburban",
            "year_built": 2000,
            "condition": "good"
        }
        self.sample_response = {
            "predicted_price": 200000.0,
            "confidence_interval": [180000.0, 220000.0],
            "features_importance": {},
            "prediction_time": "2024-01-01T12:00:00"
        }
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "model_loaded" in response.json()
    
    @patch('inference.predict_price')
    def test_predict_endpoint_success(self, mock_predict):
        """Test successful prediction."""
        # Arrange
        mock_predict.return_value = PredictionResponse(**self.sample_response)
        
        # Act
        response = self.client.post("/predict", json=self.sample_request)
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert "predicted_price" in result
        assert "confidence_interval" in result
        assert "prediction_time" in result
        mock_predict.assert_called_once()
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction with invalid input."""
        # Arrange
        invalid_request = {
            "sqft": -100,  # Invalid negative value
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "suburban",
            "year_built": 2000,
            "condition": "good"
        }
        
        # Act
        response = self.client.post("/predict", json=invalid_request)
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_fields(self):
        """Test prediction with missing required fields."""
        # Arrange
        incomplete_request = {
            "sqft": 1500.0,
            "bedrooms": 3
            # Missing other required fields
        }
        
        # Act
        response = self.client.post("/predict", json=incomplete_request)
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    @patch('inference.batch_predict')
    def test_batch_predict_endpoint_success(self, mock_batch_predict):
        """Test successful batch prediction."""
        # Arrange
        batch_request = [self.sample_request, self.sample_request]
        mock_batch_predict.return_value = [200000.0, 200000.0]
        
        # Act
        response = self.client.post("/batch-predict", json=batch_request)
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2
        mock_batch_predict.assert_called_once()
    
    def test_batch_predict_endpoint_empty_list(self):
        """Test batch prediction with empty list."""
        # Act
        response = self.client.post("/batch-predict", json=[])
        
        # Assert
        assert response.status_code == 200
        assert response.json() == []
    
    @patch('inference.predict_price')
    def test_predict_endpoint_server_error(self, mock_predict):
        """Test prediction endpoint when server error occurs."""
        # Arrange
        mock_predict.side_effect = Exception("Model loading error")
        
        # Act
        response = self.client.post("/predict", json=self.sample_request)
        
        # Assert
        assert response.status_code == 500
    
    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        # Act
        response = self.client.options("/predict")
        
        # Assert
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"


class TestSchemas:
    """Test cases for Pydantic schemas."""
    
    def test_house_prediction_request_valid(self):
        """Test valid HousePredictionRequest creation."""
        # Arrange & Act
        request = HousePredictionRequest(
            sqft=1500.0,
            bedrooms=3,
            bathrooms=2.0,
            location="suburban",
            year_built=2000,
            condition="good"
        )
        
        # Assert
        assert request.sqft == 1500.0
        assert request.bedrooms == 3
        assert request.bathrooms == 2.0
        assert request.location == "suburban"
        assert request.year_built == 2000
        assert request.condition == "good"
    
    def test_house_prediction_request_invalid_sqft(self):
        """Test HousePredictionRequest with invalid sqft."""
        # Act & Assert
        with pytest.raises(ValueError):
            HousePredictionRequest(
                sqft=-100,  # Invalid negative value
                bedrooms=3,
                bathrooms=2.0,
                location="suburban",
                year_built=2000,
                condition="good"
            )
    
    def test_house_prediction_request_invalid_bedrooms(self):
        """Test HousePredictionRequest with invalid bedrooms."""
        # Act & Assert
        with pytest.raises(ValueError):
            HousePredictionRequest(
                sqft=1500.0,
                bedrooms=0,  # Invalid: must be >= 1
                bathrooms=2.0,
                location="suburban",
                year_built=2000,
                condition="good"
            )
    
    def test_house_prediction_request_invalid_year(self):
        """Test HousePredictionRequest with invalid year."""
        # Act & Assert
        with pytest.raises(ValueError):
            HousePredictionRequest(
                sqft=1500.0,
                bedrooms=3,
                bathrooms=2.0,
                location="suburban",
                year_built=1700,  # Invalid: before 1800
                condition="good"
            )
    
    def test_prediction_response_creation(self):
        """Test PredictionResponse creation."""
        # Arrange & Act
        response = PredictionResponse(
            predicted_price=200000.0,
            confidence_interval=[180000.0, 220000.0],
            features_importance={},
            prediction_time="2024-01-01T12:00:00"
        )
        
        # Assert
        assert response.predicted_price == 200000.0
        assert response.confidence_interval == [180000.0, 220000.0]
        assert response.features_importance == {}
        assert response.prediction_time == "2024-01-01T12:00:00"


if __name__ == '__main__':
    pytest.main([__file__])
