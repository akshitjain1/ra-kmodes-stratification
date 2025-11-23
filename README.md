# 🧬 Rheumatoid Arthritis Patient Stratification using K-Modes Clustering

A clinically interpretable, patent-based machine learning system for identifying meaningful subgroups of Rheumatoid Arthritis (RA) patients using purely categorical comorbidity, lifestyle, demographic, and socioeconomic data.

Live App: https://ra-kmodes-stratification.streamlit.app/  
Patent Title: *Patient Stratification for Rheumatoid Arthritis using K-Modes Clustering on Categorical Comorbidity and Symptom Data*  
Tech Stack: Python • pandas • kmodes • Streamlit • joblib • matplotlib

---

## Overview

This project implements a categorical-only clustering pipeline based on K-Modes to discover clinically interpretable RA sub-phenotypes from categorical features only (demographics, comorbidities, lifestyle, socioeconomic, and obesity categories).

Workflow:
- Dataset preprocessing
- Categorical feature engineering
- K-Modes clustering
- Cluster interpretation
- Visualizations (bar charts, heatmaps, profiles)
- Interactive Streamlit dashboard and deployment

---

## Goals

- Identify hidden subgroups of RA patients from categorical health data.
- Produce clinically interpretable cluster profiles using dominant features.
- Provide a reusable web dashboard for researchers and clinicians.
- Keep the code modular and reproducible.

---

## Project Structure

```
ra-kmodes-stratification/
├── dashboard/
│   └── app.py                     # Streamlit app
├── src/
│   ├── config.py                  # Global paths
│   ├── data/                      # Data loading / cleaning utils
│   ├── clustering/
│   │   ├── cluster_summary.py
│   │   └── kmodes_trainer.py
│   └── visualization/
│       └── plots.py
├── notebooks/
│   └── 01_eda_ra_kmodes.ipynb
├── models/
│   └── kmodes/
│       ├── kmodes_model.pkl
│       └── category_encoder.pkl
├── data/
│   ├── raw/
│   │   └── ra_final_screened.csv
│   └── processed/
│       ├── ra_cleaned_for_kmodes.csv
│       ├── ra_encoded_for_kmodes.csv
│       └── ra_with_clusters.csv
├── requirements.txt
└── README.md
```

---

## Dataset Description

The dataset contains only categorical variables, including:

- Demographics: Gender, Age group, Race, Marital status  
- Lifestyle: Smoking status, Drinking status, Physical activity  
- Socioeconomic: Education level, Family income category  
- Anthropometric / Obesity: BMI category, BRI group (Body Roundness Index)  
- Comorbidities: Hypertension, Diabetes, Hyperlipidemia

All features adhere to the patent requirement of being categorical and clinically interpretable.

---

## Methodology

1. Preprocessing
  - Bin continuous values (e.g., Age, BRI) into meaningful categories.
  - Replace missing values with "Unknown".
  - Ensure all features are categorical strings.

2. Categorical Encoding
  - Use a custom categorical encoder and save as `category_encoder.pkl` for reuse.

3. K-Modes Clustering
  - K-Modes is used for categorical data with mismatch dissimilarity and Huang initialization.
  - Multiple K values were evaluated; K = 5 selected based on elbow in cost, clinical interpretability, and balanced cluster sizes.

4. Cluster Interpretation
  - Compute dominant category per feature, frequency tables, and clinical-style summaries.
  - Visualizations: heatmaps, bar charts, and per-cluster distributions.

5. Deployment
  - Streamlit dashboard with dataset explorer, cluster visualizations, cluster profiles, and model information.
  - Deployed on Streamlit Cloud.

---

## Visualizations

Dashboard includes:
- Cluster size distribution
- Dominant feature heatmap
- Category distributions
- Per-cluster feature profiles and frequency tables

All visualizations are implemented in `src/visualization/plots.py`.

---

## Streamlit Dashboard Features

- Home: Project summary and dataset preview  
- Dataset Explorer: Column selection, distributions, counts  
- Cluster Visualizations: Heatmaps and bar charts  
- Cluster Profiles: Per-cluster breakdown (frequency tables)  
- Model Info: K-Modes cost, iterations, and centroids

---

## Run Locally

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/ra-kmodes-stratification.git
cd ra-kmodes-stratification
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run dashboard/app.py
```

Live demo: https://ra-kmodes-stratification.streamlit.app/

---

## Requirements (example)
Add exact pins via `pip freeze > requirements.txt`. Minimal libraries:
- streamlit
- pandas
- numpy
- matplotlib
- kmodes
- joblib
- scikit-learn

---

## Limitations
- Cross-sectional dataset (not longitudinal).
- Only categorical features included; numeric clinical scores are excluded.
- The model discovers subgroups; it is not a treatment-response predictor.

---

## Future Improvements
- Add symptom-level categorical inputs
- Weight features using domain expertise
- Add cluster stability metrics
- Build predictive models on top of clusters

---

## Clinical Impact
This system can help identify high-risk RA phenotypes, stratify patient populations for research, and reveal socio-demographic and comorbidity patterns relevant for precision medicine.

---

## Author
Akshit Jain  
Machine Learning Researcher — Python • ML • Data Science • Streamlit

If you find this research useful, consider starring the repository on GitHub.
