import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_category_distribution(df: pd.DataFrame, column: str):
    """
    Plot the distribution of a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column name to plot.
    """
    plt.figure()
    counts = df[column].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_cluster_heatmap(
    df: pd.DataFrame,
    cluster_col: str,
    figsize: tuple[int, int] = (12, 6)
):
    """
    Heatmap of dominant category per feature for each cluster.

    - Data matrix: frequency (count) of the dominant category in each cluster/feature.
    - Annotation matrix: the dominant category name (string).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing categorical features + a cluster column.
    cluster_col : str
        Name of the column containing cluster labels.
    figsize : tuple of int, optional
        Figure size for the heatmap.
    """
    # features (all except cluster column)
    features = [c for c in df.columns if c != cluster_col]
    clusters = sorted(df[cluster_col].unique())

    data_matrix = []   # numeric counts / proportions
    annot_matrix = []  # category labels

    for cl in clusters:
        group = df[df[cluster_col] == cl]
        row_counts = []
        row_labels = []

        for col in features:
            vc = group[col].value_counts(normalize=False)
            if vc.empty:
                row_counts.append(0)
                row_labels.append("Unknown")
            else:
                dominant_cat = vc.index[0]
                dominant_count = vc.iloc[0]
                row_counts.append(dominant_count)
                row_labels.append(str(dominant_cat))

        data_matrix.append(row_counts)
        annot_matrix.append(row_labels)

    # Build dataframes so shapes definitely match
    index = [f"Cluster {cl}" for cl in clusters]
    data_df = pd.DataFrame(data_matrix, index=index, columns=features)
    annot_df = pd.DataFrame(annot_matrix, index=index, columns=features)

    plt.figure(figsize=figsize)
    sns.heatmap(
        data_df,
        annot=annot_df,
        fmt="",
        cmap="Blues",
        cbar=False
    )
    plt.title("Cluster Dominant Feature Heatmap")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
