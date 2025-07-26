#!/bin/bash

# MLOps Pipeline Quick Start Script
# This script sets up and runs the complete MLOps pipeline

set -e  # Exit on any error

echo "🚀 Starting MLOps Pipeline Quick Start"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check prerequisites
print_header "Checking Prerequisites"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_status "Docker found: $(docker --version | cut -d' ' -f3 | cut -d',' -f1)"
else
    print_error "Docker is required but not installed"
    exit 1
fi

# Check UV package manager
if command -v uv &> /dev/null; then
    print_status "UV package manager found"
else
    print_warning "UV package manager not found. Installing pip packages with pip instead"
    USE_PIP=true
fi

# Setup virtual environment
print_header "Setting Up Python Environment"

if [ "$USE_PIP" = true ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    print_status "Python environment setup with pip"
else
    uv venv --python python3.11
    source .venv/bin/activate
    uv pip install -r requirements.txt
    print_status "Python environment setup with UV"
fi

# Start MLflow
print_header "Starting MLflow Tracking Server"
cd deployment/mlflow
docker compose -f mlflow-docker-compose.yml up -d
print_status "MLflow started at http://localhost:5555"
cd ../..

# Wait for MLflow to be ready
print_status "Waiting for MLflow to be ready..."
sleep 10

# Phase 1: Core ML Pipeline
print_header "Phase 1: Running Core ML Pipeline"

# Data processing
print_status "Step 1: Processing raw data..."
python src/data/run_processing.py \
  --input data/raw/house_data.csv \
  --output data/processed/cleaned_house_data.csv

# Feature engineering
print_status "Step 2: Engineering features..."
python src/features/engineer.py \
  --input data/processed/cleaned_house_data.csv \
  --output data/processed/featured_house_data.csv \
  --preprocessor models/trained/preprocessor.pkl

# Model training
print_status "Step 3: Training model..."
python src/models/train_model.py \
  --config configs/model_config.yaml \
  --data data/processed/featured_house_data.csv \
  --models-dir models \
  --mlflow-tracking-uri http://localhost:5555

print_status "Phase 1 completed! Model trained and logged to MLflow"

# Start API services
print_header "Starting API Services"

# Start FastAPI
print_status "Starting FastAPI service..."
cd src/api
docker-compose up -d
cd ../..

# Start Streamlit
print_status "Starting Streamlit app..."
cd streamlit_app
docker-compose up -d
cd ..

print_status "Services started:"
print_status "  - FastAPI: http://localhost:8000 (docs: http://localhost:8000/docs)"
print_status "  - Streamlit: http://localhost:8501"

# Phase 2: Monitoring Setup (optional)
print_header "Phase 2: Setting Up Monitoring (Optional)"

read -p "Do you want to set up advanced monitoring with Grafana? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing Phase 2 monitoring dependencies..."
    
    if [ "$USE_PIP" = true ]; then
        pip install -r requirements.txt  # All dependencies are now unified
        # Optionally install dev dependencies
        read -p "Install additional development tools? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install -r requirements-dev.txt 2>/dev/null || print_warning "Dev dependencies not found or failed to install"
        fi
    else
        uv pip install -r requirements.txt  # All dependencies are now unified
        # Optionally install dev dependencies
        read -p "Install additional development tools? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            uv pip install -r requirements-dev.txt 2>/dev/null || print_warning "Dev dependencies not found or failed to install"
        fi
    fi
    
    print_status "Starting monitoring stack..."
    cd monitoring/grafana
    docker-compose up -d
    cd ../..
    
    print_status "Monitoring services started:"
    print_status "  - Grafana: http://localhost:3000 (admin/admin123)"
    print_status "  - Prometheus: http://localhost:9090"
    print_status "  - Model Metrics: http://localhost:8001/metrics"
    
    # Run monitoring demo
    print_status "Running monitoring demo..."
    python examples/phase2_monitoring_demo.py
    
    print_status "Phase 2 monitoring setup completed!"
else
    print_status "Skipping Phase 2 monitoring setup"
fi

# Final status
print_header "Setup Complete!"

echo -e "${GREEN}"
echo "🎉 MLOps Pipeline is now running!"
echo ""
echo "Access your services:"
echo "  📊 MLflow Tracking: http://localhost:5555"
echo "  🚀 FastAPI Service: http://localhost:8000"
echo "  📱 Streamlit App: http://localhost:8501"

if [[ $REPLY =~ ^[Yy]$ ]]; then
echo "  📈 Grafana Dashboard: http://localhost:3000"
echo "  🔍 Prometheus: http://localhost:9090"
fi

echo ""
echo "Next steps:"
echo "  1. Open MLflow UI to view experiment results"
echo "  2. Try the Streamlit app for interactive predictions"
echo "  3. Use the FastAPI docs to test the prediction API"
if [[ $REPLY =~ ^[Yy]$ ]]; then
echo "  4. Explore Grafana dashboards for monitoring"
fi
echo ""
echo "To stop all services:"
echo "  docker-compose down --remove-orphans"
echo ""
echo -e "${NC}"

print_status "Happy MLOps! 🚀"
