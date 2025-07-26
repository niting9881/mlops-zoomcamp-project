"""
Configuration management with secrets handling.
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "mlops_db"
    username: str = "postgres"
    password: str = ""  # Should be loaded from secrets
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "house_price_prediction"
    s3_endpoint_url: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "house_price_model"
    model_version: str = "latest"
    model_path: str = "models/trained/house_price_model.pkl"
    preprocessor_path: str = "models/trained/preprocessor.pkl"
    max_model_age_days: int = 30


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    reload: bool = False
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5


class SecretsManager:
    """Manages secrets from environment variables or secret files."""
    
    def __init__(self, secrets_file: Optional[str] = None):
        self.secrets_file = secrets_file
        self._secrets_cache = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from file if provided."""
        if self.secrets_file and Path(self.secrets_file).exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    self._secrets_cache = json.load(f)
                logger.info(f"Loaded secrets from {self.secrets_file}")
            except Exception as e:
                logger.warning(f"Failed to load secrets from {self.secrets_file}: {e}")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret value from environment or secrets file.
        
        Args:
            key: Secret key name
            default: Default value if secret not found
        
        Returns:
            Secret value or default
        """
        # Try environment variable first
        value = os.getenv(key)
        
        # Try secrets cache if not in environment
        if value is None and key in self._secrets_cache:
            value = self._secrets_cache[key]
        
        # Return default if still not found
        if value is None:
            value = default
        
        return value
    
    def set_secret(self, key: str, value: str):
        """Set secret in cache (not persistent)."""
        self._secrets_cache[key] = value


class ConfigManager:
    """Manages application configuration with secrets integration."""
    
    def __init__(self, config_file: Optional[str] = None, secrets_file: Optional[str] = None):
        self.config_file = config_file
        self.secrets_manager = SecretsManager(secrets_file)
        self._config_cache = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file if provided."""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    self._config_cache = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {self.config_file}: {e}")
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration with secrets."""
        config_data = self._config_cache.get('database', {})
        
        return DatabaseConfig(
            host=self.secrets_manager.get_secret('DB_HOST', config_data.get('host', 'localhost')),
            port=int(self.secrets_manager.get_secret('DB_PORT', str(config_data.get('port', 5432)))),
            database=self.secrets_manager.get_secret('DB_NAME', config_data.get('database', 'mlops_db')),
            username=self.secrets_manager.get_secret('DB_USERNAME', config_data.get('username', 'postgres')),
            password=self.secrets_manager.get_secret('DB_PASSWORD', config_data.get('password', ''))
        )
    
    def get_mlflow_config(self) -> MLflowConfig:
        """Get MLflow configuration with secrets."""
        config_data = self._config_cache.get('mlflow', {})
        
        return MLflowConfig(
            tracking_uri=self.secrets_manager.get_secret('MLFLOW_TRACKING_URI', 
                                                       config_data.get('tracking_uri', 'http://localhost:5000')),
            experiment_name=self.secrets_manager.get_secret('MLFLOW_EXPERIMENT_NAME',
                                                          config_data.get('experiment_name', 'house_price_prediction')),
            s3_endpoint_url=self.secrets_manager.get_secret('MLFLOW_S3_ENDPOINT_URL',
                                                          config_data.get('s3_endpoint_url')),
            aws_access_key_id=self.secrets_manager.get_secret('AWS_ACCESS_KEY_ID',
                                                            config_data.get('aws_access_key_id')),
            aws_secret_access_key=self.secrets_manager.get_secret('AWS_SECRET_ACCESS_KEY',
                                                                config_data.get('aws_secret_access_key'))
        )
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        config_data = self._config_cache.get('model', {})
        
        return ModelConfig(
            model_name=config_data.get('model_name', 'house_price_model'),
            model_version=config_data.get('model_version', 'latest'),
            model_path=config_data.get('model_path', 'models/trained/house_price_model.pkl'),
            preprocessor_path=config_data.get('preprocessor_path', 'models/trained/preprocessor.pkl'),
            max_model_age_days=config_data.get('max_model_age_days', 30)
        )
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        config_data = self._config_cache.get('api', {})
        
        return APIConfig(
            host=self.secrets_manager.get_secret('API_HOST', config_data.get('host', '0.0.0.0')),
            port=int(self.secrets_manager.get_secret('API_PORT', str(config_data.get('port', 8000)))),
            debug=config_data.get('debug', False),
            workers=int(self.secrets_manager.get_secret('API_WORKERS', str(config_data.get('workers', 1)))),
            reload=config_data.get('reload', False),
            cors_origins=config_data.get('cors_origins', ["*"])
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        config_data = self._config_cache.get('logging', {})
        
        return LoggingConfig(
            level=self.secrets_manager.get_secret('LOG_LEVEL', config_data.get('level', 'INFO')),
            format=config_data.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            file_path=self.secrets_manager.get_secret('LOG_FILE', config_data.get('file_path')),
            max_file_size_mb=config_data.get('max_file_size_mb', 10),
            backup_count=config_data.get('backup_count', 5)
        )
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations."""
        return {
            'database': self.get_database_config(),
            'mlflow': self.get_mlflow_config(),
            'model': self.get_model_config(),
            'api': self.get_api_config(),
            'logging': self.get_logging_config()
        }


# Global configuration instance
_config_manager = None


def get_config_manager(config_file: Optional[str] = None, secrets_file: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        # Default config files
        if config_file is None:
            config_file = os.getenv('CONFIG_FILE', 'config/config.json')
        
        if secrets_file is None:
            secrets_file = os.getenv('SECRETS_FILE', 'config/secrets.json')
        
        _config_manager = ConfigManager(config_file, secrets_file)
    
    return _config_manager


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config_manager().get_database_config()


def get_mlflow_config() -> MLflowConfig:
    """Get MLflow configuration."""
    return get_config_manager().get_mlflow_config()


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return get_config_manager().get_model_config()


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_config_manager().get_api_config()


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_config_manager().get_logging_config()


# Utility functions for common environment variables
def is_development() -> bool:
    """Check if running in development environment."""
    return os.getenv('ENVIRONMENT', 'development').lower() == 'development'


def is_production() -> bool:
    """Check if running in production environment."""
    env = os.getenv('ENVIRONMENT')
    return env is not None and env.lower() == 'production'


def is_staging() -> bool:
    """Check if running in staging environment."""
    env = os.getenv('ENVIRONMENT')
    return env is not None and env.lower() == 'staging'


def get_environment() -> str:
    """Get current environment."""
    return os.getenv('ENVIRONMENT', 'development').lower()
