# Rheumatoid Arthritis Patient Stratification (K-Modes Clustering)

This project implements the patented method:

**“Patient Stratification for Rheumatoid Arthritis using K-Modes Clustering on Categorical Comorbidity and Symptom Data.”**

## 🔬 Key Features
- 100% categorical feature pipeline
- K-Modes clustering (no continuous variables)
- Clinical comorbidity variables (HTN, Hyperlipidemia, Diabetes)
- BRI & BMI categorical obesity groups
- Automatic cluster profiling
- Streamlit dashboard

## 📁 Project Structure
ra-kmodes-stratification/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── clustering/
│   ├── data/
│   ├── visualization/
│   └── config.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── kmodes_model.pkl
│   └── category_encoder.pkl
│
├── notebooks/
├── reports/
│
├── requirements.txt
├── README.md
├── .gitignore
└── (venv not included)
