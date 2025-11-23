"""
Category encoder for K-Modes clustering.
Maps each categorical column to integer codes and stores mappings.
"""

import pandas as pd
from typing import Dict, Any

class CategoryEncoder:
    def __init__(self):
        self.category_maps: Dict[str, Dict[Any, int]] = {}
        self.inverse_maps: Dict[str, Dict[int, Any]] = {}

    def fit(self, df: pd.DataFrame):
        """
        Learn integer mappings for each categorical column.
        """
        for col in df.columns:
            categories = sorted(df[col].unique())
            self.category_maps[col] = {cat: i for i, cat in enumerate(categories)}
            self.inverse_maps[col] = {i: cat for cat, i in self.category_maps[col].items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned mappings to transform categorical data into integer-coded data.
        """
        encoded_df = df.copy()
        for col in df.columns:
            encoded_df[col] = df[col].map(self.category_maps[col])
        return encoded_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoder and transform data.
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert integer-coded data back to original categories.
        """
        decoded_df = df.copy()
        for col in df.columns:
            decoded_df[col] = df[col].map(self.inverse_maps[col])
        return decoded_df
