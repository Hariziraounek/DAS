import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc, mean_squared_error, r2_score)
import warnings
import os

warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global background & text ── */
    .stApp {
        background: #F8EC6C !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6A2B6F 0%, #3d1a42 100%) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f8f0ff !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #f8f0ff !important;
    }

    /* ── Main content text ── */
    .main .block-container { color: #1a0a1f; }

    /* ── Title & subtitle ── */
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #B73A5D;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center; color: #6A2B6F;
        margin-bottom: 1.5rem; font-size: 1rem;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #6A2B6F 0%, #3AA6B7 100%);
        padding: 1.2rem; border-radius: 12px; text-align: center;
        color: white !important; margin: 0.3rem;
    }
    .metric-val { font-size: 2rem; font-weight: 800; color: white !important; }
    .metric-lbl { font-size: 0.85rem; opacity: 0.85; margin-top: 0.2rem; color: white !important; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #6A2B6F;
        border-left: 5px solid #B73A5D; padding-left: 0.6rem; margin: 1.2rem 0 0.8rem;
    }

    /* ── Info boxes — DARK text explicitly forced ── */
    .insight-box {
        background: #e8f9fb;
        border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 4px solid #3AA6B7; margin: 0.5rem 0;
        color: #1a0a1f !important;
    }
    .insight-box b, .insight-box strong, .insight-box small,
    .insight-box li, .insight-box ul { color: #1a0a1f !important; }

    .warn-box {
        background: #fff0eb;
        border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 4px solid #EF695D; margin: 0.5rem 0;
        color: #1a0a1f !important;
    }
    .warn-box b, .warn-box strong { color: #1a0a1f !important; }

    /* ── Streamlit native widgets text ── */
    .stMarkdown, .stText, h1, h2, h3, h4, label, p {
        color: #1a0a1f !important;
    }
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { color: #6A2B6F !important; font-weight: 600; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid #B73A5D !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data (FIXED PATH) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # This automatically finds the folder where app.py is located
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'Heart_Disease_Prediction__2_.csv')
    
    if not os.path.exists(file_path):
        st.error(f"❌ File not found at: {file_path}. Please make sure the CSV is in the same folder as app.py!")
        st.stop()
        
    df = pd.read_csv(file_path)
    df['Target'] = (df['Heart Disease'] == 'Presence').astype(int)
    return df

@st.cache_resource
def train_models(df):
    features_clf = ['Age','Sex','Chest pain type','BP','Cholesterol','FBS over 120',
                    'EKG results','Max HR','Exercise angina','ST depression',
                    'Slope of ST','Number of vessels fluro','Thallium']
    X = df[features_clf]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_sc, y_train)
    y_pred = log_reg.predict(X_test_sc)
    y_prob = log_reg.predict_proba(X_test_sc)[:,1]

    X_lr = df[['Age','BP','Max HR','ST depression']]
    y_lr = df['Cholesterol']
    X_tr_lr, X_te_lr, y_tr_lr, y_te_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_tr_lr, y_tr_lr)
    y_pred_lr = lr_model.predict(X_te_lr)

    return {
        'log_reg': log_reg, 'scaler': scaler, 'features_clf': features_clf,
        'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob,
        'lr_model': lr_model, 'X_te_lr': X_te_lr, 'y_te_lr': y_te_lr, 'y_pred_lr': y_pred_lr
    }

df = load_data()
models = train_models(df)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 0.5rem 0 1rem 0;">
  <svg width="72" height="72" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="hg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#EF695D"/>
        <stop offset="100%" style="stop-color:#B73A5D"/>
      </linearGradient>
    </defs>
    <path d="M50 85 C50 85 10 58 10 32 C10 18 20 8 35 10 C42 11 48 16 50 21 C52 16 58 11 65 10 C80 8 90 18 90 32 C90 58 50 85 50 85Z"
          fill="url(#hg)" stroke="#F8EC6C" stroke-width="2"/>
    <path d="M30 42 L40 42 L45 32 L55 52 L60 42 L70 42"
          stroke="white" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <div style="color:#F8EC6C; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-top:4px;">HEART AI</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", [
    "🏠 Overview",
    "📦 Data Collection",
    "🔧 Preprocessing",
    "📊 EDA",
    "🤖 Modeling",
    "🔮 Predict"
])

palette = {'Absence': '#3AA6B7', 'Presence': '#B73A5D'}

# ═══════════════════════════════════════════════════════════════════════════════
if section == "🏠 Overview":
    st.markdown('<div class="main-title">❤️ Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-end ML pipeline: Collection → Preprocessing → EDA → Modeling</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">270</div><div class="metric-lbl">Total Patients</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">13</div><div class="metric-lbl">Features</div></div>', unsafe_allow_html=True)
    with c3:
        acc = accuracy_score(models['y_test'], models['y_pred'])
        st.markdown(f'<div class="metric-card"><div class="metric-val">{acc:.1%}</div><div class="metric-lbl">Model Accuracy</div></div>', unsafe_allow_html=True)
    with c4:
        fpr, tpr, _ = roc_curve(models['y_test'], models['y_prob'])
        roc_auc = auc(fpr, tpr)
        st.markdown(f'<div class="metric-card"><div class="metric-val">{roc_auc:.3f}</div><div class="metric-lbl">AUC Score</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Project Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("📦", "Data Collection", "UCI Heart Disease dataset, 270 patients, 14 features"),
        ("🔧", "Preprocessing", "Label encoding, normalization, train/test split"),
        ("📊", "EDA", "Distributions, correlations, feature analysis"),
        ("📈", "Linear Regression", "Predicting Cholesterol from clinical features"),
        ("🤖", "Logistic Regression", "Classifying Heart Disease presence/absence"),
        ("🔮", "Dashboard", "Interactive Streamlit deployment"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f'<div class="insight-box"><b>{icon} {title}</b><br><small>{desc}</small></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "📦 Data Collection":
    st.markdown('<div class="section-header">📦 Data Collection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    <b>Source:</b> UCI Machine Learning Repository — Cleveland Heart Disease Dataset<br>
    <b>Records:</b> 270 patients | <b>Features:</b> 13 clinical attributes + 1 target<br>
    <b>Task:</b> Binary classification (Presence / Absence of heart disease)
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Raw Dataset Preview")
    st.dataframe(df.drop('Target', axis=1).head(10), use_container_width=True)

    st.subheader("Feature Descriptions")
    feature_desc = pd.DataFrame({
        'Feature': ['Age','Sex','Chest pain type','BP','Cholesterol','FBS over 120',
                    'EKG results','Max HR','Exercise angina','ST depression',
                    'Slope of ST','Number of vessels fluro','Thallium','Heart Disease'],
        'Type': ['Numerical','Categorical','Categorical','Numerical','Numerical','Binary',
                 'Categorical','Numerical','Binary','Numerical','Categorical','Numerical','Categorical','Target'],
        'Description': [
            'Age of patient (years)','1=Male, 0=Female','1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic',
            'Resting blood pressure (mmHg)','Serum cholesterol (mg/dl)','Fasting blood sugar > 120 mg/dl',
            'Resting ECG results (0,1,2)','Max heart rate achieved','Exercise induced angina (1=Yes)',
            'ST depression induced by exercise','Slope of peak exercise ST (1,2,3)',
            'Number of major vessels colored by fluoroscopy','3=normal, 6=fixed defect, 7=reversible',
            'Presence / Absence of heart disease'
        ]
    })
    st.dataframe(feature_desc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🔧 Preprocessing":
    st.markdown('<div class="section-header">🔧 Data Preprocessing</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing Values Check")
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Feature','Missing Count']
        missing['Status'] = missing['Missing Count'].apply(lambda x: '✅ No missing' if x==0 else f'⚠️ {x} missing')
        st.dataframe(missing, use_container_width=True)
        st.markdown('<div class="insight-box">✅ No missing values found. Dataset is complete.</div>', unsafe_allow_html=True)

    with c2:
        st.subheader("Data Types & Stats")
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.subheader("Preprocessing Steps Applied")
    steps_info = {
        "Label Encoding": "Target variable 'Heart Disease' → binary (Presence=1, Absence=0)",
        "Feature Scaling": "StandardScaler applied to all features before Logistic Regression",
        "Train/Test Split": "80% training / 20% testing with stratification to maintain class balance",
        "No Imputation Needed": "Dataset had no missing values — no imputation required",
    }
    for step, desc in steps_info.items():
        st.markdown(f'<div class="insight-box"><b>{step}:</b> {desc}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "📊 EDA":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlations", "Feature Analysis", "Pairwise"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5,3.5))
            counts = df['Heart Disease'].value_counts()
            bars = ax.bar(counts.index, counts.values, color=['#3AA6B7','#B73A5D'], edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(val), ha='center', fontweight='bold')
            ax.set_title('Heart Disease Distribution'); ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(5,3.5))
            for label, color in palette.items():
                ax.hist(df[df['Heart Disease']==label]['Age'], bins=15, alpha=0.7, label=label, color=color)
            ax.set_title('Age Distribution by Diagnosis'); ax.legend()
            ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10,8))
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax, linewidths=0.5)
        ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
        st.pyplot(fig); plt.close()

    with tab3:
        feat = st.selectbox("Select Feature", ['Age','Cholesterol','BP','Max HR','ST depression'])
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5,3.5))
            for label, color in palette.items():
                ax.hist(df[df['Heart Disease']==label][feat], bins=15, alpha=0.7, label=label, color=color)
            ax.set_title(f'{feat} Distribution'); ax.legend(); ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(5,3.5))
            df.boxplot(column=feat, by='Heart Disease', ax=ax,
                      boxprops=dict(color='#6A2B6F'), medianprops=dict(color='#EF695D', linewidth=2))
            ax.set_title(f'{feat} Boxplot'); plt.suptitle('')
            ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig); plt.close()

    with tab4:
        fig = sns.pairplot(df[['Age','Cholesterol','BP','Max HR','Heart Disease']].copy(),
                           hue='Heart Disease', palette=palette, plot_kws={'alpha':0.5}, height=2.2)
        st.pyplot(fig.figure); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🤖 Modeling":
    st.markdown('<div class="section-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Linear Regression (Cholesterol)", "🤖 Logistic Regression (Heart Disease)"])

    with tab1:
        st.markdown("""
        <div class="insight-box">
        <b>Objective:</b> Predict Cholesterol level using Age, Blood Pressure, Max Heart Rate, and ST Depression.<br>
        This demonstrates linear regression on a continuous dependent variable.
        </div>""", unsafe_allow_html=True)

        lr = models['lr_model']
        y_te = models['y_te_lr']
        y_pr = models['y_pred_lr']
        mse = mean_squared_error(y_te, y_pr)
        r2  = r2_score(y_te, y_pr)
        rmse = np.sqrt(mse)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.metric("RMSE", f"{rmse:.2f} mg/dl")
        c3.metric("MSE", f"{mse:.2f}")

        st.markdown("""
        <div class="warn-box">
        <b>Interpretation:</b> The R² score is close to 0, indicating that Age, BP, Max HR, and ST Depression 
        have very limited linear predictive power for Cholesterol. This is a meaningful finding — 
        cholesterol in this dataset is relatively independent of other clinical measurements, 
        suggesting it may be driven by dietary and genetic factors not captured here.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y_te, y_pr, alpha=0.6, color='#3AA6B7', edgecolors='white', s=60)
            mn, mx = min(y_te.min(), y_pr.min()), max(y_te.max(), y_pr.max())
            ax.plot([mn,mx],[mn,mx],'r--', lw=2)
            ax.set_xlabel('Actual Cholesterol'); ax.set_ylabel('Predicted')
            ax.set_title(f'Actual vs Predicted\nR² = {r2:.4f}'); ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig); plt.close()
        with c2:
            residuals = y_te - y_pr
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y_pr, residuals, alpha=0.6, color='#F8EC6C', edgecolors='#EF695D', s=50)
            ax.axhline(0, color='red', linestyle='--', lw=2)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot'); ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig); plt.close()

        st.subheader("Regression Coefficients")
        coef_df = pd.DataFrame({'Feature': ['Age','BP','Max HR','ST depression'], 'Coefficient': lr.coef_})
        coef_df['Direction'] = coef_df['Coefficient'].apply(lambda x: '↑ Positive' if x > 0 else '↓ Negative')
        st.dataframe(coef_df.round(4), use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="insight-box">
        <b>Objective:</b> Classify Heart Disease presence/absence using all 13 clinical features.
        Logistic Regression with StandardScaler normalization.
        </div>""", unsafe_allow_html=True)

        acc = accuracy_score(models['y_test'], models['y_pred'])
        fpr, tpr, _ = roc_curve(models['y_test'], models['y_prob'])
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(models['y_test'], models['y_pred'])
        tn, fp, fn, tp = cm.ravel()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.2%}")
        c2.metric("AUC", f"{roc_auc:.3f}")
        c3.metric("Sensitivity (Recall)", f"{tp/(tp+fn):.2%}")
        c4.metric("Specificity", f"{tn/(tn+fp):.2%}")

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Absence','Presence'], yticklabels=['Absence','Presence'],
                        linewidths=2, cbar=False)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix\nAccuracy = {acc:.2%}')
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(fpr, tpr, color='#6A2B6F', lw=2.5, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.fill_between(fpr, tpr, alpha=0.1, color='#6A2B6F')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title('ROC Curve'); ax.legend(loc='lower right')
            ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig); plt.close()

        st.subheader("Feature Importance (Coefficients)")
        coef_series = pd.Series(models['log_reg'].coef_[0], index=models['features_clf']).sort_values()
        fig, ax = plt.subplots(figsize=(8,5))
        colors = ['#B73A5D' if c < 0 else '#3AA6B7' for c in coef_series]
        coef_series.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title('Logistic Regression Feature Importance', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="insight-box">
        <b>Interpretation:</b>
        <ul>
        <li><b>Thallium</b> and <b>Number of vessels fluoroscopy</b> are the strongest positive predictors</li>
        <li><b>Chest pain type</b> is a strong positive predictor (higher type = more disease)</li>
        <li><b>Max HR</b> is negatively correlated — higher max HR is protective</li>
        <li>The model achieves <b>85.2% accuracy</b> and <b>AUC = 0.899</b>, indicating excellent discriminative ability</li>
        </ul>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🔮 Predict":
    st.markdown('<div class="section-header">🔮 Live Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter patient data below to get a real-time prediction:")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 29, 77, 54)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        cp  = st.selectbox("Chest Pain Type", [1,2,3,4], format_func=lambda x: {1:"Typical Angina",2:"Atypical",3:"Non-Anginal",4:"Asymptomatic"}[x])
        bp  = st.slider("Blood Pressure (mmHg)", 90, 200, 130)
        chol= st.slider("Cholesterol (mg/dl)", 100, 400, 245)
    with c2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        ekg = st.selectbox("EKG Results", [0,1,2])
        hr  = st.slider("Max Heart Rate", 70, 200, 150)
        ea  = st.selectbox("Exercise Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        std = st.slider("ST Depression", 0.0, 6.2, 1.0, step=0.1)
    with c3:
        slope = st.selectbox("Slope of ST", [1,2,3])
        vessels = st.selectbox("Vessels Fluoroscopy", [0,1,2,3])
        thal  = st.selectbox("Thallium", [3,6,7], format_func=lambda x: {3:"Normal",6:"Fixed Defect",7:"Reversible"}[x])

    if st.button("🔮 Predict", use_container_width=True, type="primary"):
        input_data = np.array([[age, sex, cp, bp, chol, fbs, ekg, hr, ea, std, slope, vessels, thal]])
        input_scaled = models['scaler'].transform(input_data)
        pred = models['log_reg'].predict(input_scaled)[0]
        prob = models['log_reg'].predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ **Heart Disease: PRESENCE DETECTED** \nRisk Probability: **{prob:.1%}**")
        else:
            st.success(f"✅ **Heart Disease: ABSENCE** \nRisk Probability: **{prob:.1%}**")

        fig, ax = plt.subplots(figsize=(5,1.5))
        ax.barh(['Risk'], [prob], color='#B73A5D' if prob > 0.5 else '#3AA6B7', height=0.4)
        ax.barh(['Risk'], [1-prob], left=[prob], color='#eee', height=0.4)
        ax.set_xlim(0,1); ax.set_xlabel('Probability'); ax.set_title('Risk Score')
        ax.spines[['top','right','left']].set_visible(False)
        st.pyplot(fig); plt.close()

        st.caption("⚠️ This tool is for educational purposes only. Consult a medical professional for diagnosis.")