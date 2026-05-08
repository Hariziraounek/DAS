# ❤️ Heart Disease Prediction — End-to-End ML Project

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org)

## 📋 Project Overview

This project builds a complete machine learning pipeline to **predict heart disease** in patients using clinical data. It covers every stage of a data science workflow: data collection, preprocessing, exploratory data analysis, linear/logistic regression modeling, and an interactive Streamlit dashboard.

**Dataset:** UCI Cleveland Heart Disease Dataset  
**Records:** 270 patients | **Features:** 13 clinical attributes + 1 target  
**Target:** Binary classification — `Presence` / `Absence` of heart disease

---

## 🗂️ Project Structure

```
heart-disease-prediction/
├── app.py                      # Streamlit dashboard
├── Heart_Disease_Prediction.csv
├── README.md
└── plots/                      # Generated visualizations
    ├── 01_distribution.png
    ├── 02_age_distribution.png
    ├── 03_correlation.png
    ├── 04_scatter.png
    ├── 05_boxplots.png
    ├── 06_lr_actual_vs_pred.png
    ├── 07_residuals.png
    ├── 08_lr_coefs.png
    ├── 09_confusion_matrix.png
    ├── 10_roc_curve.png
    └── 11_feature_importance.png
```

---

## 🔬 Step 1: Data Collection

- **Source:** UCI Machine Learning Repository — Cleveland Heart Disease Dataset
- 270 patient records collected from clinical settings
- 13 features including demographic, symptomatic, and diagnostic measurements
- Target variable: `Heart Disease` → `Presence` (120 cases) / `Absence` (150 cases)

---

## 🔧 Step 2: Data Preprocessing

### Steps Applied:
| Step | Description |
|------|-------------|
| Missing Value Check | No missing values found (complete dataset) |
| Label Encoding | `Heart Disease` → binary: Presence=1, Absence=0 |
| Feature Scaling | `StandardScaler` applied before Logistic Regression |
| Train/Test Split | 80% train / 20% test with stratification |

---

## 📊 Step 3: Exploratory Data Analysis (EDA)

### Key Findings:
- The dataset is relatively balanced: 150 Absence (55.6%) vs 120 Presence (44.4%)
- **Older patients** tend to show more heart disease presence
- **Higher Max Heart Rate** is associated with *lower* disease presence (protective)
- **Chest pain type 4 (asymptomatic)** is strongly linked to disease presence
- **ST Depression** and **Number of vessels** show strong positive correlation with disease
- The correlation matrix reveals moderate multicollinearity between some features

---

## 📈 Step 4: Linear Regression — Predicting Cholesterol

### Model Setup
- **Target Variable:** Cholesterol (continuous)
- **Predictor Variables:** Age, Blood Pressure, Max Heart Rate, ST Depression
- **Purpose:** Demonstrate linear regression on a continuous dependent variable

### Results

| Metric | Value |
|--------|-------|
| **R² Score** | -0.0082 |
| **MSE** | 2799.79 |
| **RMSE** | 52.91 mg/dl |
| Intercept | 111.50 |

### Coefficients

| Feature | Coefficient | Interpretation |
|---------|-------------|---------------|
| Age | +1.4284 | Each additional year → +1.43 mg/dl cholesterol |
| BP | +0.3463 | Each 1 mmHg BP increase → +0.35 mg/dl |
| Max HR | +0.1261 | Each additional BPM → +0.13 mg/dl |
| ST Depression | -3.6426 | Each 1 unit increase → -3.64 mg/dl |

### Interpretation

> The R² score of approximately **0** (slightly negative due to test-set variance) indicates that the chosen predictors — Age, Blood Pressure, Max Heart Rate, and ST Depression — have **very limited linear predictive power** for Cholesterol levels. This is a clinically meaningful finding: serum cholesterol in this population appears to be **largely independent of other hemodynamic measurements** captured in the dataset. Cholesterol is known to be heavily influenced by dietary habits, genetic predisposition, and medication use — variables not present in this dataset. The residual plot confirms **no systematic pattern**, meaning the model assumptions are not violated; the features simply do not explain cholesterol variance well. This suggests that **a different feature set** (diet, genetics, medication) would be required to build a useful cholesterol predictor.

---

## 🤖 Step 5: Logistic Regression — Heart Disease Classification

### Model Setup
- **Target Variable:** Heart Disease (binary: 0=Absence, 1=Presence)
- **Features:** All 13 clinical features, StandardScaler normalized
- **Split:** 80% train / 20% test (stratified)

### Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **85.19%** |
| **AUC-ROC** | **0.8986** |
| Sensitivity (Recall) | 91.7% |
| Specificity | 80.0% |
| Precision (Presence) | 78.6% |
| F1-Score | 0.85 |

### Feature Importance (Logistic Coefficients)

| Feature | Effect |
|---------|--------|
| Thallium (reversible defect) | Strong positive predictor |
| Number of vessels fluoroscopy | Strong positive predictor |
| Chest pain type (asymptomatic) | Strong positive predictor |
| ST Depression | Positive predictor |
| Max Heart Rate | Negative predictor (protective) |
| Exercise Angina | Positive predictor |

### Interpretation

> The Logistic Regression model achieves **85.2% accuracy** and an **AUC of 0.899**, indicating **excellent discriminative ability** — the model can reliably distinguish between patients with and without heart disease. The high AUC (>0.85) confirms strong overall performance.  
> 
> The model is notably **sensitive (91.7% recall for Presence)**, meaning it correctly identifies most patients who have heart disease — this is clinically important, as missing a positive case is more dangerous than a false alarm.  
>
> Key risk factors identified by the model: **Thallium imaging defects**, **fluoroscopy vessel count**, and **asymptomatic chest pain** are the strongest indicators of disease presence. Conversely, **high maximum heart rate** is the strongest protective factor.

---

## 🖥️ Step 6: Streamlit Dashboard Deployment

### Installation

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

### Running the App

```bash
streamlit run app.py
```

The dashboard includes:
- 🏠 **Overview** — Project summary and KPI metrics
- 📦 **Data Collection** — Dataset description and raw data preview
- 🔧 **Preprocessing** — Missing values, scaling, encoding steps
- 📊 **EDA** — Interactive visualizations: distributions, correlations, boxplots, pairplots
- 🤖 **Modeling** — Linear and logistic regression results, ROC curve, confusion matrix
- 🔮 **Predict** — Live prediction tool with interactive sliders

---

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 🎓 Academic Context

This project was developed as part of a Data Science course. Each step follows the lab structure:
- Lab 1-2: Data Collection & Preprocessing
- Lab 3-4: EDA & Visualization
- Lab 5-6: Regression Modeling & Streamlit Deployment

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. The prediction tool does not constitute medical advice. Always consult a qualified healthcare professional for medical decisions.
