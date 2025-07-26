@echo off
REM MLOps Pipeline Quick Start Script for Windows
REM This script sets up and runs the complete MLOps pipeline

setlocal enabledelayedexpansion

echo 🚀 Starting MLOps Pipeline Quick Start
echo ======================================

REM Function to print status messages
:print_status
echo [INFO] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:print_header
echo.
echo === %~1 ===
goto :eof

REM Check prerequisites
call :print_header "Checking Prerequisites"

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Python is required but not installed"
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    call :print_status "Python found: !PYTHON_VERSION!"
)

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is required but not installed"
    pause
    exit /b 1
) else (
    call :print_status "Docker found"
)

REM Check UV package manager
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_warning "UV package manager not found. Using pip instead"
    set USE_PIP=true
) else (
    call :print_status "UV package manager found"
    set USE_PIP=false
)

REM Setup virtual environment
call :print_header "Setting Up Python Environment"

if "%USE_PIP%"=="true" (
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    call :print_status "Python environment setup with pip"
) else (
    uv venv --python python3.11
    call .venv\Scripts\activate.bat
    uv pip install -r requirements.txt
    call :print_status "Python environment setup with UV"
)

REM Start MLflow
call :print_header "Starting MLflow Tracking Server"
cd deployment\mlflow
docker-compose -f mlflow-docker-compose.yml up -d
call :print_status "MLflow started at http://localhost:5555"
cd ..\..

REM Wait for MLflow to be ready
call :print_status "Waiting for MLflow to be ready..."
timeout /t 10 /nobreak >nul

REM Phase 1: Core ML Pipeline
call :print_header "Phase 1: Running Core ML Pipeline"

REM Data processing
call :print_status "Step 1: Processing raw data..."
python src\data\run_processing.py --input data\raw\house_data.csv --output data\processed\cleaned_house_data.csv

REM Feature engineering
call :print_status "Step 2: Engineering features..."
python src\features\engineer.py --input data\processed\cleaned_house_data.csv --output data\processed\featured_house_data.csv --preprocessor models\trained\preprocessor.pkl

REM Model training
call :print_status "Step 3: Training model..."
python src\models\train_model.py --config configs\model_config.yaml --data data\processed\featured_house_data.csv --models-dir models --mlflow-tracking-uri http://localhost:5555

call :print_status "Phase 1 completed! Model trained and logged to MLflow"

REM Start API services
call :print_header "Starting API Services"

REM Start FastAPI
call :print_status "Starting FastAPI service..."
cd src\api
docker-compose up -d
cd ..\..

REM Start Streamlit
call :print_status "Starting Streamlit app..."
cd streamlit_app
docker-compose up -d
cd ..

call :print_status "Services started:"
call :print_status "  - FastAPI: http://localhost:8000 (docs: http://localhost:8000/docs)"
call :print_status "  - Streamlit: http://localhost:8501"

REM Phase 2: Monitoring Setup (optional)
call :print_header "Phase 2: Setting Up Monitoring (Optional)"

set /p SETUP_MONITORING="Do you want to set up advanced monitoring with Grafana? (y/n): "
if /i "%SETUP_MONITORING%"=="y" (
    call :print_status "Installing monitoring dependencies..."
    
    if "%USE_PIP%"=="true" (
        pip install -r requirements.txt
        REM All dependencies are now unified in requirements.txt
        set /p INSTALL_DEV="Install additional development tools? (y/n): "
        if /i "!INSTALL_DEV!"=="y" (
            pip install -r requirements-dev.txt 2>nul || call :print_warning "Dev dependencies not found or failed to install"
        )
    ) else (
        uv pip install -r requirements.txt
        REM All dependencies are now unified in requirements.txt
        set /p INSTALL_DEV="Install additional development tools? (y/n): "
        if /i "!INSTALL_DEV!"=="y" (
            uv pip install -r requirements-dev.txt 2>nul || call :print_warning "Dev dependencies not found or failed to install"
        )
    )
    
    call :print_status "Starting monitoring stack..."
    cd monitoring\grafana
    docker-compose up -d
    cd ..\..
    
    call :print_status "Monitoring services started:"
    call :print_status "  - Grafana: http://localhost:3000 (admin/admin123)"
    call :print_status "  - Prometheus: http://localhost:9090"
    call :print_status "  - Model Metrics: http://localhost:8001/metrics"
    
    REM Run monitoring demo
    call :print_status "Running monitoring demo..."
    python examples\phase2_monitoring_demo.py
    
    call :print_status "Phase 2 monitoring setup completed!"
) else (
    call :print_status "Skipping Phase 2 monitoring setup"
)

REM Final status
call :print_header "Setup Complete!"

echo.
echo 🎉 MLOps Pipeline is now running!
echo.
echo Access your services:
echo   📊 MLflow Tracking: http://localhost:5555
echo   🚀 FastAPI Service: http://localhost:8000
echo   📱 Streamlit App: http://localhost:8501

if /i "%SETUP_MONITORING%"=="y" (
    echo   📈 Grafana Dashboard: http://localhost:3000
    echo   🔍 Prometheus: http://localhost:9090
)

echo.
echo Next steps:
echo   1. Open MLflow UI to view experiment results
echo   2. Try the Streamlit app for interactive predictions
echo   3. Use the FastAPI docs to test the prediction API
if /i "%SETUP_MONITORING%"=="y" (
    echo   4. Explore Grafana dashboards for monitoring
)
echo.
echo To stop all services:
echo   docker-compose down --remove-orphans
echo.

call :print_status "Happy MLOps! 🚀"
pause
