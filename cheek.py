# check_packages.py
import sys

print("Python version:", sys.version)

packages = [
    "numpy",
    "pandas", 
    "matplotlib",
    "seaborn",
    "sklearn",
    "librosa",
    "streamlit",
    "joblib"
]

for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {package}: {version}")
    except ImportError:
        print(f"❌ {package}: NOT INSTALLED")

print("\n🎉 All checks completed!")