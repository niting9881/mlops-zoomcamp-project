"""
Test runner script for the MLOps project.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_type="all", coverage=True, verbose=False):
    """
    Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'data', 'all')
        coverage: Whether to run with coverage
        verbose: Whether to run with verbose output
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test paths based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "data":
        cmd.append("tests/data/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml"
        ])
    
    # Add verbose if requested
    if verbose:
        cmd.append("-v")
    
    # Run tests
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode

def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="Run tests for MLOps project")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "data", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Run without coverage"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run with verbose output"
    )
    
    args = parser.parse_args()
    
    # Set environment variable for testing
    os.environ["ENVIRONMENT"] = "test"
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        coverage=not args.no_coverage,
        verbose=args.verbose
    )
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
