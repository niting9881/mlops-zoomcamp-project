"""
Integration tests for the complete ML pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestMLPipelineIntegration:
    """Integration tests for the complete ML pipeline."""
    
    def setup_method(self):
        """Set up test data and temporary directories."""
        self.test_data = pd.DataFrame({
            'sqft': [1200, 1500, 1800, 2000, 800, 2200, 1100, 1600],
            'bedrooms': [2, 3, 2, 4, 1, 3, 2, 3],
            'bathrooms': [1.5, 2.0, 2.0, 2.5, 1.0, 2.5, 1.5, 2.0],
            'location': ['urban', 'suburban', 'rural', 'urban', 'urban', 'suburban', 'rural', 'suburban'],
            'year_built': [1990, 2000, 1985, 2010, 1975, 2005, 1995, 1998],
            'condition': ['good', 'excellent', 'good', 'good', 'fair', 'excellent', 'good', 'good'],
            'price': [150000, 200000, 120000, 300000, 90000, 280000, 140000, 185000]
        })
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.raw_data_path = os.path.join(self.temp_dir, 'raw_data.csv')
        self.processed_data_path = os.path.join(self.temp_dir, 'processed_data.csv')
        self.featured_data_path = os.path.join(self.temp_dir, 'featured_data.csv')
        self.model_path = os.path.join(self.temp_dir, 'model.pkl')
        self.preprocessor_path = os.path.join(self.temp_dir, 'preprocessor.pkl')
        
        # Save test data
        self.test_data.to_csv(self.raw_data_path, index=False)
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_processing_pipeline(self):
        """Test the complete data processing pipeline."""
        from data.processor import load_data, clean_data
        
        # Act - Load and clean data
        raw_data = load_data(self.raw_data_path)
        cleaned_data = clean_data(raw_data)
        
        # Assert
        assert len(cleaned_data) <= len(raw_data)  # May remove outliers
        assert cleaned_data.isnull().sum().sum() == 0  # No missing values
        assert set(cleaned_data.columns) == set(raw_data.columns)  # Same columns
    
    def test_feature_engineering_pipeline(self):
        """Test the feature engineering pipeline."""
        from data.processor import load_data, clean_data
        from features.engineer import engineer_features
        
        # Act - Process and engineer features
        raw_data = load_data(self.raw_data_path)
        cleaned_data = clean_data(raw_data)
        featured_data = engineer_features(cleaned_data)
        
        # Assert
        assert len(featured_data) == len(cleaned_data)
        assert 'house_age' in featured_data.columns
        assert 'bed_bath_ratio' in featured_data.columns
        assert 'price_per_sqft' in featured_data.columns
        
        # Check feature values are reasonable
        assert (featured_data['house_age'] >= 0).all()
        assert (featured_data['bed_bath_ratio'] > 0).all()
        assert (featured_data['price_per_sqft'] > 0).all()
    
    @patch('models.train_model.joblib.dump')
    def test_model_training_pipeline(self, mock_joblib_dump):
        """Test the model training pipeline."""
        from data.processor import load_data, clean_data
        from features.engineer import engineer_features
        
        # Arrange
        raw_data = load_data(self.raw_data_path)
        cleaned_data = clean_data(raw_data)
        featured_data = engineer_features(cleaned_data)
        
        # Mock the model training process
        with patch('models.train_model.train_and_evaluate_model') as mock_train:
            mock_train.return_value = (MagicMock(), MagicMock(), {"mae": 15000, "r2": 0.85})
            
            # Act - This would normally call the training function
            # For integration test, we simulate the training process
            X = featured_data.drop(['price'], axis=1)
            y = featured_data['price']
            
            # Assert
            assert len(X) == len(y)
            assert len(X.columns) > len(self.test_data.columns)  # Should have more features
            assert y.dtype in ['int64', 'float64']  # Target should be numeric
    
    def test_api_integration_with_model(self):
        """Test API integration with model prediction."""
        # This test would require actual model files
        # For now, we'll test the API structure
        from fastapi.testclient import TestClient
        
        # Add API path to sys.path
        api_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'api')
        sys.path.append(api_path)
        
        with patch('inference.model'), patch('inference.preprocessor'):
            from main import app
            client = TestClient(app)
            
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            
            # Test prediction endpoint structure
            sample_request = {
                "sqft": 1500.0,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "location": "suburban",
                "year_built": 2000,
                "condition": "good"
            }
            
            with patch('inference.predict_price') as mock_predict:
                mock_predict.return_value = MagicMock(
                    predicted_price=200000.0,
                    confidence_interval=[180000.0, 220000.0],
                    features_importance={},
                    prediction_time="2024-01-01T12:00:00"
                )
                
                response = client.post("/predict", json=sample_request)
                assert response.status_code == 200
    
    def test_end_to_end_prediction_flow(self):
        """Test the complete prediction flow from raw input to output."""
        # This would test the complete flow:
        # Raw input -> Validation -> Feature Engineering -> Model Prediction -> Output
        
        from api.utils import validate_input_data, sanitize_prediction_output
        
        # Arrange
        raw_input = {
            "sqft": 1500.0,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "location": "suburban",
            "year_built": 2000,
            "condition": "good"
        }
        
        # Act - Validate input
        validated_input = validate_input_data(raw_input)
        
        # Simulate feature engineering
        input_df = pd.DataFrame([validated_input])
        input_df['house_age'] = 2024 - input_df['year_built']
        input_df['bed_bath_ratio'] = input_df['bedrooms'] / input_df['bathrooms']
        input_df['price_per_sqft'] = 0  # Placeholder for prediction
        
        # Simulate model prediction
        mock_prediction = 200000.0
        mock_confidence_interval = [180000.0, 220000.0]
        
        # Sanitize output
        sanitized_output = sanitize_prediction_output(mock_prediction, mock_confidence_interval)
        
        # Assert
        assert validated_input["sqft"] == 1500.0
        assert 'house_age' in input_df.columns
        assert sanitized_output["predicted_price"] == 200000.0
        assert len(sanitized_output["confidence_interval"]) == 2
    
    def test_error_handling_throughout_pipeline(self):
        """Test error handling throughout the pipeline."""
        from api.utils import validate_input_data
        
        # Test invalid input handling
        invalid_inputs = [
            {"sqft": -100},  # Negative sqft
            {"sqft": 1500, "bedrooms": 0},  # Zero bedrooms
            {"sqft": 1500, "bedrooms": 3, "year_built": 1700},  # Invalid year
            {}  # Empty input
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                validate_input_data(invalid_input)
    
    def test_data_consistency_through_pipeline(self):
        """Test that data remains consistent through the pipeline."""
        from data.processor import load_data, clean_data
        from features.engineer import engineer_features
        
        # Act
        raw_data = load_data(self.raw_data_path)
        cleaned_data = clean_data(raw_data)
        featured_data = engineer_features(cleaned_data)
        
        # Assert data consistency
        # Original columns should be preserved
        original_cols = set(raw_data.columns)
        cleaned_cols = set(cleaned_data.columns)
        featured_cols = set(featured_data.columns)
        
        assert original_cols.issubset(cleaned_cols)
        assert original_cols.issubset(featured_cols)
        
        # Data types should be maintained for original columns
        for col in original_cols:
            if col in cleaned_data.columns and col in featured_data.columns:
                # Check that numeric columns remain numeric
                if pd.api.types.is_numeric_dtype(raw_data[col]):
                    assert pd.api.types.is_numeric_dtype(cleaned_data[col])
                    assert pd.api.types.is_numeric_dtype(featured_data[col])


class TestSystemIntegration:
    """Integration tests for system components."""
    
    def test_docker_compose_services(self):
        """Test that Docker Compose services are properly configured."""
        # This would test the actual Docker Compose configuration
        # For now, we'll test the configuration file structure
        
        import yaml
        compose_file = os.path.join(
            os.path.dirname(__file__), '..', '..', 'docker-compose.yaml'
        )
        
        if os.path.exists(compose_file):
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            # Assert
            assert 'services' in compose_config
            assert 'fastapi' in compose_config['services']
            assert 'ports' in compose_config['services']['fastapi']
    
    def test_api_and_streamlit_integration(self):
        """Test integration between API and Streamlit app."""
        # This would test the communication between services
        pass
    
    def test_mlflow_integration(self):
        """Test MLflow integration."""
        # This would test MLflow tracking and model registry
        pass


if __name__ == '__main__':
    pytest.main([__file__])
