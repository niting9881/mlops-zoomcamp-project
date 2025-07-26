"""
Phase 2 Monitoring Requirements Installation Script

This script installs additional dependencies required for Phase 2 MLOps monitoring features.
Note: As of the latest version, all dependencies are included in the main requirements.txt file.
This script is kept for backward compatibility.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install Phase 2 monitoring requirements from main requirements file."""
    
    print("Installing MLOps Pipeline dependencies (including Phase 2 monitoring)...")
    print("=" * 60)
    print("Note: All dependencies are now unified in requirements.txt")
    print("=" * 60)
    
    try:
        # Install from main requirements file
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Successfully installed all dependencies!")
        print("\nInstalled packages include:")
        print("  - Core ML libraries (scikit-learn, xgboost, pandas, numpy)")
        print("  - Web frameworks (FastAPI, Streamlit, uvicorn)")
        print("  - Monitoring tools (prometheus-client, grafana-api, psutil)")
        print("  - Data validation (great-expectations, pandera)")
        print("  - Experiment tracking (MLflow)")
        print("  - Testing tools (pytest, coverage)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("Error output:")
        print(e.stderr)
        return False
    
    return True

def install_dev_requirements():
    """Install additional development dependencies."""
    
    print("\nInstalling additional development dependencies...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Successfully installed development dependencies!")
        print("  - Advanced monitoring (elasticsearch, redis)")
        print("  - Development tools (black, flake8, mypy)")
        print("  - Documentation tools (sphinx, mkdocs)")
        print("  - Testing tools (pytest extensions)")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error installing dev requirements: {e}")
        print("This is optional - continuing without dev dependencies")
        return False
    except FileNotFoundError:
        print("⚠️ requirements-dev.txt not found - skipping dev dependencies")
        return False
    
if __name__ == "__main__":
    print("🚀 MLOps Pipeline Dependencies Installer")
    print("=" * 60)
    
    # Install main requirements
    if install_requirements():
        print("\n✅ Core dependencies installed successfully!")
        
        # Ask user if they want dev dependencies
        try:
            install_dev = input("\nInstall additional development dependencies? (y/N): ").lower().strip()
            if install_dev in ['y', 'yes']:
                install_dev_requirements()
        except KeyboardInterrupt:
            print("\n\nSkipping development dependencies...")
    
    print("\n" + "=" * 60)
    print("🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Start MLflow: cd deployment/mlflow && docker-compose up -d")
    print("2. Run the pipeline: ./quick-start.sh (Linux/Mac) or quick-start.bat (Windows)")
    print("3. Try the monitoring demo: python examples/phase2_monitoring_demo.py")
    print("=" * 60)
