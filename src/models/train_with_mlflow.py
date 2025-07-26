# mlflow_tracking.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Set the tracking URI - this could be a local directory or a remote server
mlflow.set_tracking_uri("http://localhost:5555")

# Set the experiment
mlflow.set_experiment("house-price-prediction")

# Load the data
data = pd.read_csv('data/processed/featured_house_data.csv')
X = data.drop('price', axis=1)
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters to try
n_estimators_options = [50, 100, 200]
max_depth_options = [None, 10, 20]

# Try different parameter combinations
for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        # Start a new MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            
            # Train the model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate and log metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log the model
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            print(f"n_estimators={n_estimators}, max_depth={max_depth}: MAE={mae:.2f}, R2={r2:.2f}")
