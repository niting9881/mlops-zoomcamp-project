#!/usr/bin/env python3
"""
Comprehensive System Functionality Test Suite
Tests all components of the MLOps system without depending on the API
"""

import requests
import json
import time
import sys
import os
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {"SUCCESS": GREEN, "ERROR": RED, "WARNING": YELLOW, "INFO": BLUE}
    color = colors.get(status, BLUE)
    print(f"{color}[{status}] {message}{RESET}")

def test_service_health(url, service_name, timeout=5):
    """Test if a service is responding"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print_status(f"{service_name} is healthy (Status: {response.status_code})", "SUCCESS")
            return True
        else:
            print_status(f"{service_name} returned status {response.status_code}", "WARNING")
            return False
    except requests.exceptions.RequestException as e:
        print_status(f"{service_name} is not responding: {str(e)}", "ERROR")
        return False

def test_service_endpoint(url, service_name, expected_content=None):
    """Test a specific service endpoint"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            content = response.text
            if expected_content and expected_content in content:
                print_status(f"{service_name} endpoint working correctly", "SUCCESS")
                return True
            elif not expected_content:
                print_status(f"{service_name} endpoint accessible", "SUCCESS")
                return True
            else:
                print_status(f"{service_name} endpoint accessible but content unexpected", "WARNING")
                return False
        else:
            print_status(f"{service_name} endpoint returned status {response.status_code}", "ERROR")
            return False
    except requests.exceptions.RequestException as e:
        print_status(f"{service_name} endpoint error: {str(e)}", "ERROR")
        return False

def test_model_files():
    """Test if model files exist and are accessible"""
    model_files = [
        "models/trained/house_price_model.pkl",
        "models/trained/preprocessor.pkl"
    ]
    
    all_good = True
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print_status(f"Model file {model_file} exists ({size} bytes)", "SUCCESS")
        else:
            print_status(f"Model file {model_file} missing", "ERROR")
            all_good = False
    
    return all_good

def test_data_files():
    """Test if data files exist"""
    data_files = [
        "data/raw/house_data.csv",
        "data/processed/cleaned_house_data.csv"
    ]
    
    all_good = True
    for data_file in data_files:
        if os.path.exists(data_file):
            size = os.path.getsize(data_file)
            print_status(f"Data file {data_file} exists ({size} bytes)", "SUCCESS")
        else:
            print_status(f"Data file {data_file} missing", "WARNING")
    
    return all_good

def test_configuration():
    """Test if configuration files are accessible"""
    config_files = [
        "config/config.json",
        "config/config_manager.py"
    ]
    
    all_good = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print_status(f"Config file {config_file} exists", "SUCCESS")
        else:
            print_status(f"Config file {config_file} missing", "ERROR")
            all_good = False
    
    return all_good

def main():
    """Run comprehensive system tests"""
    print_status("Starting MLOps System Functionality Tests", "INFO")
    print("=" * 60)
    
    # Test results storage
    results = {}
    
    # 1. Test Core Services
    print_status("Testing Core Services...", "INFO")
    services = {
        "Streamlit": "http://localhost:8501",
        "MLflow": "http://localhost:5555",
        "Prometheus": "http://localhost:9090",
        "Grafana": "http://localhost:3000",
        "Model Exporter": "http://localhost:8001/metrics"
    }
    
    for service_name, url in services.items():
        results[service_name] = test_service_health(url, service_name)
    
    print()
    
    # 2. Test Specific Endpoints
    print_status("Testing Specific Endpoints...", "INFO")
    endpoints = {
        "Streamlit App": ("http://localhost:8501", "House Price Prediction"),
        "MLflow UI": ("http://localhost:5555", "MLflow"),
        "Prometheus Metrics": ("http://localhost:9090/metrics", "prometheus_build_info"),
        "Grafana Dashboard": ("http://localhost:3000/login", "Grafana"),
        "Model Metrics": ("http://localhost:8001/metrics", "model_prediction_requests_total")
    }
    
    for endpoint_name, (url, expected_content) in endpoints.items():
        results[f"{endpoint_name}_endpoint"] = test_service_endpoint(url, endpoint_name, expected_content)
    
    print()
    
    # 3. Test File System Components
    print_status("Testing File System Components...", "INFO")
    results["model_files"] = test_model_files()
    results["data_files"] = test_data_files()
    results["config_files"] = test_configuration()
    
    print()
    
    # 4. Test Monitoring Stack Integration
    print_status("Testing Monitoring Stack Integration...", "INFO")
    try:
        # Test if Prometheus can scrape model metrics
        prom_targets_url = "http://localhost:9090/api/v1/targets"
        response = requests.get(prom_targets_url, timeout=10)
        if response.status_code == 200:
            targets_data = response.json()
            active_targets = len([t for t in targets_data.get('data', {}).get('activeTargets', []) if t.get('health') == 'up'])
            print_status(f"Prometheus has {active_targets} active targets", "SUCCESS")
            results["prometheus_targets"] = active_targets > 0
        else:
            print_status("Could not retrieve Prometheus targets", "WARNING")
            results["prometheus_targets"] = False
    except Exception as e:
        print_status(f"Error testing Prometheus integration: {str(e)}", "ERROR")
        results["prometheus_targets"] = False
    
    print()
    
    # 5. Summary Report
    print_status("Test Summary Report", "INFO")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {GREEN}{passed_tests}{RESET}")
    print(f"Failed: {RED}{failed_tests}{RESET}")
    print(f"Success Rate: {GREEN if success_rate >= 80 else YELLOW if success_rate >= 60 else RED}{success_rate:.1f}%{RESET}")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = GREEN if result else RED
        print(f"  {color}{status}{RESET} - {test_name}")
    
    print("\n" + "=" * 60)
    
    if success_rate >= 80:
        print_status("✅ System is functioning well! Most components are operational.", "SUCCESS")
        
        print("\n🚀 Quick Access URLs:")
        print("  • Streamlit App: http://localhost:8501")
        print("  • MLflow Tracking: http://localhost:5555")
        print("  • Grafana Dashboard: http://localhost:3000 (admin/admin)")
        print("  • Prometheus: http://localhost:9090")
        
    elif success_rate >= 60:
        print_status("⚠️  System is partially functional. Some components need attention.", "WARNING")
    else:
        print_status("❌ System has significant issues. Multiple components are not working.", "ERROR")
    
    return success_rate >= 80

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("Test interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", "ERROR")
        sys.exit(1)
