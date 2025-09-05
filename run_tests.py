#!/usr/bin/env python3
"""Simple script to run tests that can be called by Poetry scripts."""

import subprocess
import sys
import os


def main():
    """Run pytest with the appropriate arguments."""
    # Try to use the virtual environment if available
    venv_python = os.path.join("test_venv", "bin", "python")
    
    if os.path.exists(venv_python):
        cmd = [venv_python, "-m", "pytest"] + sys.argv[1:]
    else:
        # Fallback to system pytest (may fail in externally managed environments)
        cmd = ["python", "-m", "pytest"] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, cwd=".")
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("pytest not found. Please install pytest with:")
        print("  pip install pytest pytest-cov pytest-mock")
        print("Or use the provided virtual environment:")
        print("  source test_venv/bin/activate && python -m pytest")
        sys.exit(1)


if __name__ == "__main__":
    main()