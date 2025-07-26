# Import required libraries
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import yaml
import logging
from mlflow.tracking import MlflowClient
import platform
import sklearn
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open('configs/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logger.info(f"Loaded configuration: {config}")

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment(config['model']['name'])

# Load data
logger.info("Loading data...")
data = pd.read_csv('data/processed/featured_house_data.csv')

# Split features and target
X = data.drop('price', axis=1)
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Define models to evaluate
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror')
}

# Define evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'model': model
    }

# Train and evaluate models
results = {}

with mlflow.start_run(run_name="model_comparison"):
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        with mlflow.start_run(run_name=name, nested=True):
            # Log model parameters
            params = model.get_params()
            mlflow.log_params(params)
            
            # Train and evaluate
            evaluation = evaluate_model(model, X_train, X_test, y_train, y_test)
            results[name] = evaluation
            
            # Log metrics
            mlflow.log_metrics({
                'mae': evaluation['mae'],
                'mse': evaluation['mse'],
                'rmse': evaluation['rmse'],
                'r2': evaluation['r2']
            })
            
            # Log model
            mlflow.sklearn.log_model(model, name)
            
            logger.info(f"{name} - MAE: {evaluation['mae']:.2f}, RMSE: {evaluation['rmse']:.2f}, R²: {evaluation['r2']:.4f}")

# Identify best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
logger.info(f"Best model: {best_model_name} with R²: {results[best_model_name]['r2']:.4f}")


# Hyperparameter tuning for best model
logger.info(f"Performing hyperparameter tuning for {best_model_name}...")

if best_model_name == 'RandomForest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'GradientBoosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
else:  # LinearRegression has no hyperparameters to tune
    param_grid = {}

if param_grid:
    with mlflow.start_run(run_name=f"{best_model_name}_tuning"):
        grid_search = GridSearchCV(
            best_model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluate tuned model
        tuned_model = grid_search.best_estimator_
        y_pred = tuned_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        })
        
        # Log model
        mlflow.sklearn.log_model(tuned_model, "tuned_model")
        
        logger.info(f"Tuned {best_model_name} - Best params: {grid_search.best_params_}")
        logger.info(f"Tuned {best_model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        # Save final model
        logger.info("Saving final model...")
        joblib.dump(tuned_model, f"models/trained/house_price_model.pkl")
        
        # Log final model test metrics
        logger.info(f"Final Model Test Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        mlflow.log_metrics({'final_mae': mae, 'final_rmse': rmse, 'final_r2': r2})
        
        # Register the model in MLflow Model Registry
        model_name = "house_price_model"  # Name of the model in the registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"

        logger.info(f"Registering the model '{model_name}' to MLflow Model Registry...")
        client = MlflowClient()

        # Register the model
        try:
            registered_model = client.create_registered_model(model_name)
            logger.info(f"Created new registered model: {model_name}")
        except mlflow.exceptions.RestException:
            logger.info(f"Model '{model_name}' already exists in the registry.")

        # Create a new model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )
        logger.info(f"Registered model version: {model_version.version}")

        # Transition the model to "Staging" or "Production" (optional)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f"Model version {model_version.version} transitioned to 'Staging'.")

        # Add a description to the registered model
        description = (
            f"Model for predicting house prices. "
            f"Trained on dataset: 'engineered_features.csv'. "
            f"Best model: {best_model_name}. "
            f"Hyperparameters: {grid_search.best_params_ if param_grid else 'Default parameters'}."
        )
        client.update_registered_model(
            name=model_name,
            description=description
        )
        logger.info(f"Added description to the registered model: {description}")

        # Log dependencies as tags
        dependencies = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,  # Use the correct alias
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }

        for key, value in dependencies.items():
            client.set_registered_model_tag(model_name, key, value)
        logger.info(f"Logged dependencies: {dependencies}")

        # Add additional tags to the registered model
        tags = {
            "dataset_version": "v1.0",
            "environment": "staging",
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "model_owner": "data_science_team",
            "model_type": best_model_name,
        }

        for key, value in tags.items():
            client.set_registered_model_tag(model_name, key, value)
        logger.info(f"Added tags to the registered model: {tags}")

        # Add metrics as tags to the registered model
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

        for key, value in metrics.items():
            client.set_registered_model_tag(model_name, f"metric_{key}", value)
        logger.info(f"Logged metrics as tags: {metrics}")

        # Log preprocessing steps
        preprocessing_steps = "Applied feature engineering: house_age, price_per_sqft, bed_bath_ratio. One-hot encoding for categorical variables."
        client.set_registered_model_tag(model_name, "preprocessing_steps", preprocessing_steps)
        logger.info(f"Logged preprocessing steps: {preprocessing_steps}")

        # Log model URI
        client.set_registered_model_tag(model_name, "model_uri", model_uri)
        logger.info(f"Logged model URI: {model_uri}")
else:
    # Save best model without tuning
    logger.info("Saving final model...")
    joblib.dump(best_model, f"models/trained/house_price_model.pkl")

logger.info("Training complete!")