"""
Configuration module for RA K-Modes stratification project.
"""

from pathlib import Path

# Base project directory (adjust if needed)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models directory
MODELS_DIR = BASE_DIR / "models" / "kmodes"

# Random seed for reproducibility
RANDOM_STATE = 42
