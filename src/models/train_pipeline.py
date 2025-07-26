import argparse
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
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train house price prediction models.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    parser.add_argument("--data", type=str, required=True, help="Path to the processed dataset CSV file.")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained models.")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI (e.g., http://localhost:5555).")
    return parser.parse_args()

# Main function
def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration: {config}")

    # Set up MLflow if enabled
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(config['model']['name'])
        logger.info(f"MLflow tracking URI set to: {args.mlflow_tracking_uri}")

    # Load data
    logger.info("Loading data...")
    data = pd.read_csv(args.data)

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

    # TODO : Pass the hyperparams to XGBoost': xgb.XGBRegressor(objective='reg:squarederror')

    # Define evaluation function for Hyperparameter tuning

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

    with mlflow.start_run(run_name="model_comparison") if args.mlflow_tracking_uri else nullcontext():
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            with mlflow.start_run(run_name=name, nested=True) if args.mlflow_tracking_uri else nullcontext():
                # Log model parameters
                if args.mlflow_tracking_uri:
                    params = model.get_params()
                    mlflow.log_params(params)
                
                # Train and evaluate
                evaluation = evaluate_model(model, X_train, X_test, y_train, y_test)
                results[name] = evaluation
                
                # Log metrics
                if args.mlflow_tracking_uri:
                    mlflow.log_metrics({
                        'mae': evaluation['mae'],
                        'mse': evaluation['mse'],
                        'rmse': evaluation['rmse'],
                        'r2': evaluation['r2']
                    })
                
                # Log model
                if args.mlflow_tracking_uri:
                    mlflow.sklearn.log_model(model, name)
                
                logger.info(f"{name} - MAE: {evaluation['mae']:.2f}, RMSE: {evaluation['rmse']:.2f}, R²: {evaluation['r2']:.4f}")

    # Identify best model
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    logger.info(f"Best model: {best_model_name} with R²: {results[best_model_name]['r2']:.4f}")

    # Save the best model
    model_path = f"{args.models_dir}/trained/house_price_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model to: {model_path}")

# Entry point
if __name__ == "__main__":
    args = parse_args()
    main(args)