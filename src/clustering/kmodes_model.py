"""
K-Modes clustering wrapper for RA patient stratification.
"""

from typing import Optional
import pandas as pd
from kmodes.kmodes import KModes


def build_kmodes_model(
    n_clusters: int,
    init_method: str = "Huang",
    random_state: int = 42
) -> KModes:
    """
    Create a K-Modes model with Huang initialization.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    init_method : str
        Initialization method ('Huang' or 'Cao').
    random_state : int
        Random seed.

    Returns
    -------
    KModes
        Configured K-Modes model instance.
    """
    model = KModes(
        n_clusters=n_clusters,
        init=init_method,
        n_init=5,
        verbose=0,
        random_state=random_state
    )
    return model
