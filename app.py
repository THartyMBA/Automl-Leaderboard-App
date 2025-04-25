# automl_leaderboard_app.py
"""
AutoML Leaderboard Studio  🚀
────────────────────────────────────────────────────────────────────────────
Upload ANY tabular CSV, pick a target, and let the app:

1. Auto-detect numeric vs. categorical features.
2. Train a small ensemble of off-the-shelf classifiers:
      • Logistic Regression
      • Gradient Boosting
      • Random Forest
      • LightGBM (uses gradient boosting trees; skipped if lib not available)
3. Score each model on Accuracy, F1, and ROC-AUC.
4. Display a sortable leaderboard.
5. Show a SHAP feature-importance bar plot for the *champion* model.
6. Let you download the champion model (`model.pkl`) **and** a scored CSV.

This is a proof-of-concept demo.  For production-grade AutoML pipelines,
model tracking, and CI/CD, contact me → **drtomharty.com/bio**.
"""

# ───────────────────────────────────────────────────── imports ─────────────────
import io, pickle, os, warnings
import pandas as pd
import numpy as np
import streamlit as st
import shap
import plotly.graph_objects as go

from sklearn.compose        import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report
)

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import GradientBoostingClassifier, RandomForestClassifier

# optional LightGBM
try:
    from lightgbm            import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    warnings.filterwarnings("ignore", "LightGBM")


# ───────────────────────────────────────────────── config / helpers ────────────
st.set_page_config(page_title="AutoML Leaderboard Studio", layout="wide")
shap.initjs()

def build_preprocessor(df: pd.DataFrame, target_col: str):
    num_cols = df.drop(columns=[target_col]).select_dtypes(include="number").columns
    cat_cols = df.drop(columns=[target_col]).select_dtypes(exclude="number").columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

def get_models():
    models = {
        "Logistic Regression":       LogisticRegression(max_iter=2000, n_jobs=-1),
        "Gradient Boosting":         GradientBoostingClassifier(),
        "Random Forest":             RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    return models

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def train_single_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]  # assumes binary
    pred  = (proba > 0.5).astype(int)
    return {
        "name": name,
        "model": model,
        "accuracy": accuracy_score(y_val, pred),
        "f1":       f1_score(y_val, pred, zero_division=0),
        "roc_auc":  roc_auc_score(y_val, proba)
    }

def shap_bar_plot(pipe, X_val):
    # Use 2k sample for speed
    background = shap.sample(X_val, min(2000, X_val.shape[0]))
    try:
        explainer = shap.Explainer(pipe["clf"], background)
        shap_values = explainer(pipe["clf"].booster_.predict(background, raw_score=True) 
                                if hasattr(pipe["clf"], "booster_") else background)
    except Exception:
        # generic explainer fallback (slower)
        explainer = shap.Explainer(pipe.predict_proba, background)
        shap_values = explainer(X_val)
    shap_fig = shap.plots.bar(shap_values, show=False)
    return shap_fig

# ─────────────────────────────────────────────────── UI ────────────────────────
st.title("🏆 AutoML Leaderboard Studio")

# Universal demo notice
st.info(
    "🔔 **Demo Notice**  \n"
    "This is a lightweight proof-of-concept. For production-grade AutoML "
    "with experiment tracking, CI/CD, and governance, [contact me](https://drtomharty.com/bio).",
    icon="💡"
)

file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if not file:
    st.stop()

df = load_csv(file)
st.subheader("Preview")
st.dataframe(df.head())

# Target & settings
target_col = st.selectbox("🎯 Choose target column (binary classification only)", df.columns)
test_size  = st.slider("Validation split %", 0.1, 0.4, 0.2, 0.05)
metric_to_sort = st.selectbox("Order leaderboard by", ["roc_auc", "accuracy", "f1"])

if st.button("🚀 Train Leaderboard"):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    pre = build_preprocessor(df, target_col)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    leaderboard = []
    for name, base_model in get_models().items():
        pipe = Pipeline([("pre", pre), ("clf", base_model)])
        with st.spinner(f"Training {name}…"):
            res = train_single_model(name, pipe, X_train, y_train, X_val, y_val)
        leaderboard.append(res)

    lb_df = pd.DataFrame(leaderboard).sort_values(metric_to_sort, ascending=False).reset_index(drop=True)
    best_row = lb_df.iloc[0]
    champ_name = best_row["name"]
    champ_model = next(r["model"] for r in leaderboard if r["name"] == champ_name)

    st.subheader("Leaderboard")
    st.dataframe(lb_df.style.format({"accuracy": "{:.3f}", "f1": "{:.3f}", "roc_auc": "{:.3f}"}))

    # SHAP plot
    st.subheader(f"🔍 Feature Importance – {champ_name}")
    st.caption("Bars show mean absolute SHAP value (impact on prediction).")

    # Extract transformed validation set for SHAP
    X_val_trans = champ_model.named_steps["pre"].transform(X_val)
    shap_fig = shap_bar_plot(champ_model, X_val_trans)
    st.pyplot(shap_fig)

    # Score full data for download
    full_proba = champ_model.predict_proba(X)[:, 1]
    scored_df = df.copy()
    scored_df["pred_proba"] = full_proba

    # Downloads
    st.subheader("Downloads")
    st.download_button(
        label="⬇️ Scored CSV",
        data=scored_df.to_csv(index=False).encode(),
        file_name="scored_data.csv",
        mime="text/csv"
    )
    st.download_button(
        label="💾 Champion model (.pkl)",
        data=pickle.dumps(champ_model),
        file_name="model.pkl",
        mime="application/octet-stream"
    )
