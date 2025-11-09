import io
import os
import textwrap
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

MODEL_PATH = "alz_model.pkl"
RANDOM_STATE = 42

st.set_page_config(page_title="Alzheimer's Detection Demo", page_icon="🧠", layout="centered")

# ---------- Helpers ----------
def load_sample_data() -> pd.DataFrame:
    # Tiny, illustrative dataset (not clinical!). Columns:
    # age, MMSE, CDR, EDUC, eTIV, gender, target (0=Control, 1=Alzheimer's)
    csv = textwrap.dedent("""\
    age,MMSE,CDR,EDUC,eTIV,gender,target
    72,18,1,10,1500,M,1
    68,26,0.5,12,1470,F,1
    75,12,2,8,1520,F,1
    62,29,0,16,1555,M,0
    58,30,0,15,1490,F,0
    66,27,0.5,12,1510,M,1
    70,24,0.5,10,1485,F,1
    60,28,0,14,1540,M,0
    64,30,0,18,1575,F,0
    73,20,1,9,1460,M,1
    69,25,0.5,13,1505,F,1
    55,30,0,17,1530,F,0
    """)
    return pd.read_csv(io.StringIO(csv))

def make_preprocessor(df: pd.DataFrame):
    num_features = ["age", "MMSE", "CDR", "EDUC", "eTIV"]
    cat_features = ["gender"]

    transformers = []
    if num_features:
        from sklearn.impute import SimpleImputer
        transformers.append(("num", SimpleImputer(strategy="median"), num_features))
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def train_model(df: pd.DataFrame):
    df = df.copy()
    required = ["age", "MMSE", "CDR", "EDUC", "eTIV", "gender", "target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[["age", "MMSE", "CDR", "EDUC", "eTIV", "gender"]]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    pre = make_preprocessor(df)
    # Handle potential imbalance
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights = {cls: w for cls, w in zip(classes, cw)}

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight=weights
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None,
        "test_size": int(X_test.shape[0])
    }

    return pipe, metrics

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def predict_one(model, age, mmse, cdr, educ, etiv, gender):
    X = pd.DataFrame([{
        "age": age,
        "MMSE": mmse,
        "CDR": cdr,
        "EDUC": educ,
        "eTIV": etiv,
        "gender": gender
    }])
    prob = float(model.predict_proba(X)[0, 1])
    label = int(prob >= 0.5)
    return label, prob

def batch_predict(model, df: pd.DataFrame):
    needed = ["age", "MMSE", "CDR", "EDUC", "eTIV", "gender"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"CSV is missing column: {c}")
    probs = model.predict_proba(df[needed])[:, 1]
    labels = (probs >= 0.5).astype(int)
    out = df.copy()
    out["alz_prob"] = probs
    out["alz_pred"] = labels
    return out

# ---------- UI ----------
st.title("🧠 Alzheimer's Detection (Demo App)")
st.caption("Educational demo — not for medical use.")

tabs = st.tabs(["Predict", "Batch CSV", "Train / Retrain", "About"])

# --- Predict tab ---
with tabs[0]:
    st.subheader("Single Prediction")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=40, max_value=100, value=68, step=1)
        mmse = st.number_input("MMSE (0–30)", min_value=0, max_value=30, value=26, step=1)
        cdr = st.select_slider("CDR (0, 0.5, 1, 2, 3)", options=[0.0, 0.5, 1.0, 2.0, 3.0], value=0.5)
    with col2:
        educ = st.number_input("Years of Education", min_value=0, max_value=25, value=12, step=1)
        etiv = st.number_input("eTIV (approx.)", min_value=1200, max_value=2000, value=1500, step=5)
        gender = st.selectbox("Gender", options=["M", "F"], index=1)

    model = load_model()
    if model is None:
        st.info("No saved model found. Using a quick model trained on the built-in sample data.")
        sample_df = load_sample_data()
        model, _ = train_model(sample_df)

    if st.button("Predict"):
        try:
            label, prob = predict_one(model, age, mmse, cdr, educ, etiv, gender)
            st.metric("Predicted Class (0=Control, 1=AD)", value=label)
            st.progress(min(max(prob, 0.0), 1.0))
            st.write(f"Estimated probability of Alzheimer's: **{prob:.2%}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Batch CSV tab ---
with tabs[1]:
    st.subheader("Batch Predictions from CSV")
    st.write("Required columns: `age, MMSE, CDR, EDUC, eTIV, gender`")
    st.caption("Tip: download a template below and fill it.")

    # Template CSV
    template = pd.DataFrame({
        "age": [72, 62],
        "MMSE": [18, 29],
        "CDR": [1.0, 0.0],
        "EDUC": [10, 16],
        "eTIV": [1500, 1555],
        "gender": ["M", "F"]
    })
    st.download_button("Download CSV template", template.to_csv(index=False), file_name="alz_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            model = load_model()
            if model is None:
                st.info("No saved model found. Using a quick model trained on the built-in sample data.")
                sample_df = load_sample_data()
                model, _ = train_model(sample_df)
            out = batch_predict(model, df_in)
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False), file_name="alz_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# --- Train / Retrain tab ---
with tabs[2]:
    st.subheader("Train / Retrain the Model")
    st.write("Upload a labeled CSV with columns: `age, MMSE, CDR, EDUC, eTIV, gender, target` "
             "(where `target` is 0 for Control and 1 for Alzheimer's).")

    st.caption("If you don't have data, use the built-in sample to see how it works.")
    c1, c2 = st.columns(2)
    with c1:
        train_choice = st.radio("Choose training data:", ["Upload CSV", "Use built-in sample"], index=1)
    with c2:
        save_opt = st.checkbox("Save trained model to disk", value=True)

    train_df = None
    if train_choice == "Upload CSV":
        train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train_csv")
        if train_file:
            try:
                train_df = pd.read_csv(train_file)
                st.success(f"Loaded {train_df.shape[0]} rows.")
                st.dataframe(train_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
    else:
        train_df = load_sample_data()
        st.info("Using built-in sample dataset (for demo only).")

    if st.button("Train now") and train_df is not None:
        try:
            model, metrics = train_model(train_df)
            if save_opt:
                save_model(model)
                st.success(f"Model trained and saved to `{MODEL_PATH}`.")
            else:
                st.success("Model trained (not saved).")

            st.write("**Evaluation on held-out set:**")
            st.write(f"ROC-AUC: {metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "ROC-AUC: n/a")
            st.write("Confusion Matrix ([[TN, FP],[FN, TP]]):")
            st.code(str(metrics["confusion_matrix"]))
            st.write("Classification Report:")
            # Show as a small table
            rep_df = pd.DataFrame(metrics["classification_report"]).T
            st.dataframe(rep_df, use_container_width=True)
        except Exception as e:
            st.error(f"Training failed: {e}")

# --- About tab ---
with tabs[3]:
    st.subheader("About this app")
    st.write(
        "This is a minimal educational demo showing how to build a small ML pipeline for Alzheimer's "
        "classification using features like Age, MMSE, CDR, Education, and eTIV. "
        "The included data are synthetic/toy and this tool **must not** be used for any medical decision."
    )
    st.markdown(
        "- **Inputs**: age, MMSE (0–30), CDR (0–3), years of education, eTIV, gender\n"
        "- **Model**: Random Forest with simple preprocessing (imputation + one-hot for gender)\n"
        "- **Target**: 0 = Control, 1 = Alzheimer's\n"
    )
    st.caption("Always consult qualified clinicians for diagnosis. This demo is for learning purposes only.")
