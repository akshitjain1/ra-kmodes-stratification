"""
Cluster evaluation helpers:
- Cost (K-Modes cost function)
- Placeholder for categorical silhouette-style metrics
"""

import numpy as np
from kmodes.kmodes import KModes


def get_kmodes_cost(model: KModes) -> float:
    """
    Return the cost (sum of mismatches) of a fitted K-Modes model.

    Parameters
    ----------
    model : KModes
        Fitted K-Modes model.

    Returns
    -------
    float
        Clustering cost.
    """
    return float(model.cost_)
