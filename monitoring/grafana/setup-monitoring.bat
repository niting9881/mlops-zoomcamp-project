@echo off
REM MLOps Grafana Monitoring Setup Script for Windows

echo 🚀 Setting up MLOps Grafana Monitoring Infrastructure
echo ============================================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Docker is not running. Please start Docker first.
    exit /b 1
)

echo ✅ Docker is running

REM Navigate to monitoring directory
cd monitoring\grafana

echo 📦 Building and starting monitoring stack...

REM Stop any existing containers
docker-compose down

REM Pull latest images
echo 🔄 Pulling latest images...
docker-compose pull

REM Build and start services
echo 🏗️  Building and starting services...
docker-compose up -d --build

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service status
echo 🔍 Checking service status...
docker-compose ps

REM Wait for Grafana to be ready
echo ⏳ Waiting for Grafana to start...
set /a timeout=120
set /a counter=0
:wait_grafana
curl -s http://localhost:3000 >nul 2>&1
if not errorlevel 1 goto grafana_ready
if %counter% geq %timeout% (
    echo ❌ Timeout waiting for Grafana to start
    exit /b 1
)
echo Waiting for Grafana... (%counter%/%timeout%)
timeout /t 5 /nobreak >nul
set /a counter+=5
goto wait_grafana

:grafana_ready
echo ✅ Grafana is ready!

REM Wait for Prometheus to be ready
echo ⏳ Waiting for Prometheus to start...
set /a timeout=60
set /a counter=0
:wait_prometheus
curl -s http://localhost:9090 >nul 2>&1
if not errorlevel 1 goto prometheus_ready
if %counter% geq %timeout% (
    echo ❌ Timeout waiting for Prometheus to start
    exit /b 1
)
echo Waiting for Prometheus... (%counter%/%timeout%)
timeout /t 5 /nobreak >nul
set /a counter+=5
goto wait_prometheus

:prometheus_ready
echo ✅ Prometheus is ready!

REM Check if model exporter is running
echo 🔍 Checking model exporter status...
curl -s http://localhost:8001/metrics >nul 2>&1
if not errorlevel 1 (
    echo ✅ Model exporter is ready!
) else (
    echo ⚠️  Model exporter may take a few more seconds to be ready
)

echo.
echo 🎉 MLOps Monitoring Setup Complete!
echo ============================================================
echo.
echo 📊 Access your monitoring services:
echo    🔵 Grafana Dashboard: http://localhost:3000
echo       Username: admin
echo       Password: admin123
echo.
echo    🟡 Prometheus: http://localhost:9090
echo    🟢 Model Metrics: http://localhost:8001/metrics
echo.
echo 📈 Available Dashboards:
echo    • MLOps Model Monitoring Dashboard
echo    • Data Drift Monitoring Dashboard
echo.
echo 🔧 Useful Commands:
echo    • View logs: docker-compose logs -f
echo    • Stop services: docker-compose down
echo    • Restart services: docker-compose restart
echo.
echo 📚 Next Steps:
echo    1. Access Grafana at http://localhost:3000
echo    2. Login with admin/admin123
echo    3. Navigate to dashboards to view ML monitoring
echo    4. Make some predictions to see metrics in action
echo.
