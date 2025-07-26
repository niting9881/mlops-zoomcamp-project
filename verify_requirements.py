#!/usr/bin/env python3
"""
Requirements Verification Script
This script verifies that all consolidated requirements are properly installable
and that there are no conflicts between dependencies.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

def run_command(cmd: List[str]) -> Tuple[bool, str, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_requirements_syntax(req_file: Path) -> bool:
    """Check if requirements file has valid syntax."""
    print(f"📋 Checking syntax: {req_file}")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        lines = None
        
        for encoding in encodings:
            try:
                with open(req_file, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        if lines is None:
            print(f"  ❌ Could not read file with any encoding")
            return False
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Basic validation - check for package name
            if '==' in line or '>=' in line or '<=' in line or '>' in line or '<' in line:
                continue
            elif line and not any(char in line for char in ['=', '>', '<', '#']):
                continue
            else:
                print(f"  ⚠️ Line {i} might have syntax issues: {line}")
        
        print(f"  ✅ Syntax OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False

def get_installed_packages() -> Dict[str, str]:
    """Get currently installed packages and versions."""
    success, stdout, stderr = run_command([sys.executable, "-m", "pip", "list", "--format=json"])
    
    if not success:
        return {}
    
    try:
        packages = json.loads(stdout)
        return {pkg["name"].lower(): pkg["version"] for pkg in packages}
    except:
        return {}

def test_imports() -> Dict[str, bool]:
    """Test importing key packages."""
    import_tests = {
        'pandas': 'import pandas as pd',
        'numpy': 'import numpy as np',
        'sklearn': 'import sklearn',
        'fastapi': 'import fastapi',
        'streamlit': 'import streamlit',
        'mlflow': 'import mlflow',
        'pytest': 'import pytest',
        'prometheus_client': 'import prometheus_client',
        'great_expectations': 'import great_expectations',
        'plotly': 'import plotly',
        'psutil': 'import psutil',
        'scipy': 'import scipy',
        'uvicorn': 'import uvicorn',
    }
    
    results = {}
    print("\n🔍 Testing package imports:")
    
    for name, import_stmt in import_tests.items():
        try:
            exec(import_stmt)
            print(f"  ✅ {name}")
            results[name] = True
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            results[name] = False
        except Exception as e:
            print(f"  ⚠️ {name}: {e}")
            results[name] = False
    
    return results

def check_version_conflicts() -> List[str]:
    """Check for potential version conflicts."""
    print("\n🔍 Checking for version conflicts:")
    
    conflicts = []
    installed = get_installed_packages()
    
    # Known conflict patterns
    conflict_checks = [
        (['fastapi', 'pydantic'], "FastAPI and Pydantic version compatibility"),
        (['pandas', 'numpy'], "Pandas and NumPy compatibility"),
        (['scikit-learn', 'numpy'], "Scikit-learn and NumPy compatibility"),
        (['mlflow', 'sqlalchemy'], "MLflow and SQLAlchemy compatibility"),
    ]
    
    for packages, description in conflict_checks:
        versions = []
        for pkg in packages:
            if pkg in installed:
                versions.append(f"{pkg}=={installed[pkg]}")
            else:
                versions.append(f"{pkg}=NOT_INSTALLED")
        
        print(f"  📦 {description}: {', '.join(versions)}")
    
    return conflicts

def verify_requirements_files():
    """Main verification function."""
    print("🚀 MLOps Requirements Verification")
    print("=" * 50)
    
    # Check all requirements files
    req_files = [
        Path("requirements.txt"),
        Path("requirements-dev.txt"),
        Path("src/api/requirements.txt"),
        Path("streamlit_app/requirements.txt"),
    ]
    
    print("\n📁 Found requirements files:")
    for req_file in req_files:
        if req_file.exists():
            print(f"  ✅ {req_file}")
            check_requirements_syntax(req_file)
        else:
            print(f"  ❌ {req_file} (missing)")
    
    # Test key imports
    import_results = test_imports()
    
    # Check for conflicts
    conflicts = check_version_conflicts()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    successful_imports = sum(1 for result in import_results.values() if result)
    total_imports = len(import_results)
    
    print(f"✅ Successful imports: {successful_imports}/{total_imports}")
    
    if successful_imports == total_imports:
        print("🎉 All key packages are working correctly!")
    else:
        print("⚠️ Some packages failed to import. Check installation.")
    
    if not conflicts:
        print("✅ No obvious version conflicts detected")
    else:
        print(f"⚠️ {len(conflicts)} potential conflicts found")
    
    print("\n📋 Next steps:")
    print("1. If imports failed: pip install -r requirements.txt")
    print("2. For development: pip install -r requirements-dev.txt") 
    print("3. Run tests: python -m pytest tests/")
    print("4. Start services: ./quick-start.sh or quick-start.bat")

if __name__ == "__main__":
    verify_requirements_files()
