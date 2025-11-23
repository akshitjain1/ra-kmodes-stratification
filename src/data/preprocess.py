"""
Preprocessing functions for categorical RA data:
- Handle missing values (Unknown category)
- Map raw clinical values to consistent categories
"""

import pandas as pd


def handle_missing_as_unknown(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Replace missing values in specified columns with 'Unknown'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list of str
        Columns to process.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'Unknown' filled for missing values.
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna("Unknown")
    return df
