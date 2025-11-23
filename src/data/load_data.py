"""
Functions for loading Rheumatoid Arthritis datasets
for K-Modes clustering.
"""

from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_DIR


def load_ra_dataset(filename: str) -> pd.DataFrame:
    """
    Load the RA dataset from the raw data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file in the raw data folder.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    return pd.read_csv(file_path)
