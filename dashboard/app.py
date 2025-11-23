# dashboard/app.py

import sys
from pathlib import Path

# --- Make project root importable as a package ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Project imports (from our own modules)
from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.clustering.cluster_summary import summarize_clusters

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------------------
# Global Streamlit config
# ------------------------------
st.set_page_config(
    page_title="RA Stratification Dashboard",
    layout="wide"
)

CLUSTER_COL = "Cluster"  # name of the cluster label column in ra_with_clusters.csv


# ------------------------------
# Cached loaders
# ------------------------------
@st.cache_data
def load_clustered_data() -> pd.DataFrame:
    """Load processed dataset with cluster labels."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "ra_with_clusters.csv")
    # Basic safety check
    if CLUSTER_COL not in df.columns:
        raise ValueError(
            f"Expected a '{CLUSTER_COL}' column in ra_with_clusters.csv but did not find it."
        )
    return df


@st.cache_resource
def load_models():
    """Load trained K-Modes model and category encoder."""
    km = joblib.load(MODELS_DIR / "kmodes_model.pkl")
    enc = joblib.load(MODELS_DIR / "category_encoder.pkl")
    return km, enc


@st.cache_data
def load_cluster_summaries(df: pd.DataFrame):
    """Precompute cluster summaries for dashboard use."""
    summaries = summarize_clusters(
        df.drop(columns=[CLUSTER_COL]),
        df[CLUSTER_COL]
    )
    return summaries


# Actually load data and models once
df = load_clustered_data()
km_model, encoder = load_models()
cluster_summaries = load_cluster_summaries(df)


# ==========================================================
# Helper plotting functions (dashboard-specific)
# ==========================================================
def st_plot_category_distribution(df: pd.DataFrame, column: str):
    """Plot categorical distribution and render in Streamlit."""
    counts = df[column].value_counts()

    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {column}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)


def st_plot_cluster_sizes(df: pd.DataFrame):
    """Bar chart of cluster sizes."""
    counts = df[CLUSTER_COL].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Cluster Sizes")
    fig.tight_layout()
    st.pyplot(fig)


def st_plot_cluster_dominant_heatmap(df: pd.DataFrame):
    """
    Plot heatmap of dominant category per (cluster, feature).

    Color encodes an arbitrary numeric code; annotation shows the category label,
    which is what the clinician actually cares about.
    """
    features = [c for c in df.columns if c != CLUSTER_COL]

    # Build a matrix of "codes" and a matrix of labels
    mode_labels = []
    code_matrix = []

    # For each feature, create a mapping category -> integer code (for colors)
    feature_category_maps = {}
    for col in features:
        cats = df[col].astype(str).unique().tolist()
        feature_category_maps[col] = {cat: i for i, cat in enumerate(sorted(cats))}

    for cluster_id, group in df.groupby(CLUSTER_COL):
        row_labels = []
        row_codes = []
        for col in features:
            # Most frequent category in this cluster for this feature
            mode_cat = group[col].value_counts().idxmax()
            row_labels.append(str(mode_cat))
            row_codes.append(feature_category_maps[col][str(mode_cat)])
        mode_labels.append(row_labels)
        code_matrix.append(row_codes)

    mode_labels_df = pd.DataFrame(mode_labels, columns=features)
    code_df = pd.DataFrame(code_matrix, columns=features)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(code_df.values, aspect="auto")

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticks(range(len(code_df)))
    ax.set_yticklabels([f"Cluster {i}" for i in code_df.index])

    # Annotate with actual category labels
    for i in range(code_df.shape[0]):
        for j in range(code_df.shape[1]):
            ax.text(
                j,
                i,
                mode_labels_df.iloc[i, j],
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title("Cluster Dominant Feature Heatmap")
    fig.tight_layout()
    st.pyplot(fig)


# ==========================================================
# App Navigation
# ==========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset Explorer",
        "Cluster Visualizations",
        "Cluster Profiles",
        "Model Information",
    ],
)


# ==========================================================
# PAGE 1 — HOME
# ==========================================================
if page == "Home":
    st.title("Rheumatoid Arthritis Patient Stratification")
    st.subheader("K-Modes Clustering • Categorical Comorbidity & Symptom Data")

    st.write(
        """
This dashboard implements the pipeline described in the patent:

> **“Patient Stratification for Rheumatoid Arthritis using K-Modes Clustering on Categorical Comorbidity and Symptom Data.”**

The system performs **unsupervised clustering** of RA patients using only categorical variables:

- Demographics  
- Lifestyle patterns  
- Socioeconomic indicators  
- Cardiometabolic comorbidities  
- BMI / BRI-based obesity categories  

Cluster labels in this dashboard are produced by a **K-Modes model** trained on a categorical matrix.
"""
    )

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Basic Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Patients", len(df))
    with col2:
        st.metric("Number of Features (excluding cluster label)", len(df.columns) - 1)
    with col3:
        st.metric("Number of Clusters (K)", int(km_model.n_clusters))


# ==========================================================
# PAGE 2 — DATASET EXPLORER
# ==========================================================
elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")

    st.write(
        """
Explore the processed RA dataset that was used as input to the K-Modes algorithm.
You can:
- View the full table  
- Select specific columns  
- Inspect category distributions  
"""
    )

    st.write("### Full Dataset")
    st.dataframe(df)

    st.write("### Select Columns to View")
    selected_cols = st.multiselect(
        "Choose columns",
        df.columns.tolist(),
        default=df.columns.tolist(),
    )
    st.dataframe(df[selected_cols])

    st.write("---")
    st.write("### Category Distribution")

    cat_col = st.selectbox(
        "Select a categorical column",
        df.columns.tolist(),
        index=df.columns.tolist().index(CLUSTER_COL) if CLUSTER_COL in df.columns else 0,
    )

    st.write(f"#### Distribution of **{cat_col}**")
    st_plot_category_distribution(df, cat_col)

    st.write("---")
    st.write("### Value Counts")
    st.write(df[cat_col].value_counts())


# ==========================================================
# PAGE 3 — CLUSTER VISUALIZATIONS
# ==========================================================
elif page == "Cluster Visualizations":
    st.title("🧩 Cluster Visualizations")

    st.write(
        """
This section helps visually inspect the discovered RA patient clusters:

- **Cluster Sizes** — how many patients fall into each subgroup  
- **Dominant Feature Heatmap** — for each cluster and feature, the most common category  
"""
    )

    st.write("### Cluster Sizes")
    st_plot_cluster_sizes(df)

    st.write("---")
    st.write("### Cluster Dominant Feature Heatmap")

    st.info(
        "Each cell shows the **most frequent category** for that feature within the cluster. "
        "Color is just a code; the text annotation is what matters clinically."
    )

    st_plot_cluster_dominant_heatmap(df)


# ==========================================================
# PAGE 4 — CLUSTER PROFILES
# ==========================================================
elif page == "Cluster Profiles":
    st.title("🩺 Cluster Profiles")

    st.write(
        """
Here you can examine **detailed categorical summaries** for each RA cluster.

For every cluster:
- Each feature shows a frequency table (category vs. proportion).
- This matches the interpretation style described in the patent (mode-based cluster summaries).
"""
    )

    cluster_ids = sorted(cluster_summaries.keys())
    selected_cluster = st.selectbox(
        "Select a cluster to inspect",
        cluster_ids,
        format_func=lambda x: f"Cluster {x}",
    )

    st.write(f"## Cluster {selected_cluster} — Feature-wise Category Proportions")

    summary_for_cluster = cluster_summaries[selected_cluster]

    # Show overall size of this cluster
    cluster_size = (df[CLUSTER_COL] == selected_cluster).sum()
    st.write(f"Number of patients in this cluster: **{cluster_size}**")

    # Display each feature in an expander
    for feature_name, freq_df in summary_for_cluster.items():
        with st.expander(f"{feature_name}"):
            # freq_df: index = category, column 'proportion'
            display_df = freq_df.copy()
            display_df["percentage"] = (display_df["proportion"] * 100).round(2)
            st.dataframe(display_df)


# ==========================================================
# PAGE 5 — MODEL INFORMATION
# ==========================================================
elif page == "Model Information":
    st.title("🧠 Model Information — K-Modes Clustering Engine")

    st.write(
        """
This page summarizes the **K-Modes model** and the categorical feature space used
for Rheumatoid Arthritis patient stratification.
"""
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Clusters (K)", int(km_model.n_clusters))
    with col2:
        st.metric("K-Modes Cost", float(km_model.cost_))
    with col3:
        st.metric("Iterations to Converge", int(km_model.n_iter_))

    st.write("### Categorical Features Used")
    feature_cols = [c for c in df.columns if c != CLUSTER_COL]
    st.write(feature_cols)

    st.write("---")
    st.write("### Cluster Centroids (Modes)")

    # km_model.cluster_centroids_ has shape (K, num_features), same order as features used.
    centroids = pd.DataFrame(
        km_model.cluster_centroids_,
        columns=feature_cols,
    )
    centroids.index = [f"Cluster {i}" for i in range(len(centroids))]
    st.dataframe(centroids)

    st.write(
        """
Each row represents the **mode (most frequent category)** for that cluster
across all input features. This is consistent with the patent's emphasis
on *mode-based, interpretable RA sub-phenotypes*.
"""
    )
