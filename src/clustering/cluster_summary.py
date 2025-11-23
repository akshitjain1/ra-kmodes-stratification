"""
Cluster summary utilities:
- Compute dominant patterns per cluster
- Generate frequency tables for interpretation
"""

import pandas as pd


def summarize_clusters(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    max_categories: int = 5
) -> dict:
    """
    Create simple summaries for each cluster:
    - Category frequency tables for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Categorical feature matrix.
    cluster_labels : pd.Series or array-like
        Cluster assignments for each row.
    max_categories : int
        Max number of top categories to keep per feature.

    Returns
    -------
    dict
        Mapping: cluster_id -> { feature_name -> frequency_table_df }
    """
    df = df.copy()
    df["cluster"] = cluster_labels

    summaries: dict = {}
    for cluster_id, group in df.groupby("cluster"):
        cluster_summary = {}
        for col in df.columns:
            if col == "cluster":
                continue
            freq = group[col].value_counts(normalize=True).head(max_categories)
            freq_df = freq.rename("proportion").to_frame()
            cluster_summary[col] = freq_df
        summaries[int(cluster_id)] = cluster_summary

    return summaries
