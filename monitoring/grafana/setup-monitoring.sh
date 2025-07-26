#!/bin/bash

# MLOps Grafana Monitoring Setup Script

echo "🚀 Setting up MLOps Grafana Monitoring Infrastructure"
echo "=" * 60

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Navigate to monitoring directory
cd monitoring/grafana

echo "📦 Building and starting monitoring stack..."

# Stop any existing containers
docker-compose down

# Pull latest images
echo "🔄 Pulling latest images..."
docker-compose pull

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

# Wait for Grafana to be ready
echo "⏳ Waiting for Grafana to start..."
timeout=120
counter=0
while ! curl -s http://localhost:3000 > /dev/null; do
    if [ $counter -ge $timeout ]; then
        echo "❌ Timeout waiting for Grafana to start"
        exit 1
    fi
    echo "Waiting for Grafana... ($counter/$timeout)"
    sleep 5
    counter=$((counter + 5))
done

echo "✅ Grafana is ready!"

# Wait for Prometheus to be ready
echo "⏳ Waiting for Prometheus to start..."
timeout=60
counter=0
while ! curl -s http://localhost:9090 > /dev/null; do
    if [ $counter -ge $timeout ]; then
        echo "❌ Timeout waiting for Prometheus to start"
        exit 1
    fi
    echo "Waiting for Prometheus... ($counter/$timeout)"
    sleep 5
    counter=$((counter + 5))
done

echo "✅ Prometheus is ready!"

# Check if model exporter is running
echo "🔍 Checking model exporter status..."
if curl -s http://localhost:8001/metrics > /dev/null; then
    echo "✅ Model exporter is ready!"
else
    echo "⚠️  Model exporter may take a few more seconds to be ready"
fi

echo ""
echo "🎉 MLOps Monitoring Setup Complete!"
echo "=" * 60
echo ""
echo "📊 Access your monitoring services:"
echo "   🔵 Grafana Dashboard: http://localhost:3000"
echo "      Username: admin"
echo "      Password: admin123"
echo ""
echo "   🟡 Prometheus: http://localhost:9090"
echo "   🟢 Model Metrics: http://localhost:8001/metrics"
echo ""
echo "📈 Available Dashboards:"
echo "   • MLOps Model Monitoring Dashboard"
echo "   • Data Drift Monitoring Dashboard"
echo ""
echo "🔧 Useful Commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop services: docker-compose down"
echo "   • Restart services: docker-compose restart"
echo ""
echo "📚 Next Steps:"
echo "   1. Access Grafana at http://localhost:3000"
echo "   2. Login with admin/admin123"
echo "   3. Navigate to dashboards to view ML monitoring"
echo "   4. Make some predictions to see metrics in action"
echo ""
