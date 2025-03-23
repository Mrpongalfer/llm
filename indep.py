#!/usr/bin/env python

import subprocess
import sys
import os

def create_requirements_file():
    """Create a requirements.txt file with all required packages."""
    requirements = [
        "torch",
        "transformers",
        "scikit-learn",
        "optuna",
        "requests",
        "beautifulsoup4",
        "numpy",
        "dataclasses"
    ]
    
    with open("requirements.txt", "w") as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("✅ Created requirements.txt file")

def install_requirements():
    """Install packages from requirements.txt file."""
    print("📦 Installing required packages...")
    
    # Check if virtual environment should be created
    create_venv = input("Create virtual environment? (y/n): ").strip().lower() == 'y'
    
    if create_venv:
        venv_name = input("Enter virtual environment name (default: 'venv'): ").strip()
        if not venv_name:
            venv_name = "venv"
        
        print(f"🔨 Creating virtual environment: {venv_name}")
        subprocess.run([sys.executable, "-m", "venv", venv_name])
        
        # Determine the pip path based on the OS
        if sys.platform == "win32":
            pip_path = os.path.join(venv_name, "Scripts", "pip")
        else:
            pip_path = os.path.join(venv_name, "bin", "pip")
        
        # Upgrade pip
        subprocess.run([pip_path, "install", "--upgrade", "pip"])
        
        # Install requirements
        result = subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    else:
        # Use current Python interpreter's pip
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    if result.returncode == 0:
        print("✅ All packages installed successfully!")
    else:
        print("❌ Error installing packages. Please check the output above.")

def check_installation():
    """Verify all packages can be imported."""
    print("🔍 Verifying package installation...")
    
    packages_to_check = [
        "torch", "transformers", "sklearn", "optuna", 
        "requests", "bs4", "numpy", "dataclasses"
    ]
    
    all_successful = True
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {str(e)}")
            all_successful = False
    
    if all_successful:
        print("\n🎉 All packages verified successfully!")
    else:
        print("\n⚠️ Some packages could not be imported. Please check the errors above.")

def main():
    print("=== Python Package Installer ===")
    create_requirements_file()
    install_requirements()
    check_installation()
    print("\nInstallation process completed.")

if __name__ == "__main__":
    main()
