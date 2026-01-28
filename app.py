# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# -------------------------
# Load trained models & encoders
# -------------------------
severity_model = joblib.load("models/severity_model.pkl")
icu_model = joblib.load("models/icu_model.pkl")
scaler = joblib.load("models/scaler.pkl")
sex_encoder = joblib.load("models/sex_encoder.pkl")
severity_encoder = joblib.load("models/severity_encoder.pkl")

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Advanced Clinical Decision Support System (CDSS)",
    layout="centered"
)

st.title("🩺 Clinical Decision Support System (CDSS)")
st.markdown(
    """
    This system predicts **disease severity** and **ICU admission risk**.
    You can input patient data manually or upload a CSV file.
    SHAP explainability is included to show feature impact.
    """
)

# -------------------------
# Input selection
# -------------------------
st.sidebar.title("Input Method")
input_mode = st.sidebar.radio(
    "Choose Input Type",
    ["Manual Entry", "Upload CSV File"]
)

# -------------------------
# Manual Input Function
# -------------------------
def manual_input():
    age = st.slider("Age", 18, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    comorbidities = st.slider("Number of Comorbidities", 0, 5, 1)
    crp = st.slider("CRP (mg/L)", 5, 200, 50)
    wbc = st.slider("WBC (cells/µL)", 4000, 16000, 8000)
    fever = st.selectbox("Fever", [0, 1])
    oxygen = st.slider("Oxygen Saturation (%)", 70, 100, 95)

    # Encode sex
    sex_encoded = sex_encoder.transform([sex])[0]

    df = pd.DataFrame([[
        age, sex_encoded, comorbidities, crp, wbc, fever, oxygen
    ]], columns=[
        "Age", "Sex", "Comorbidities", "CRP", "WBC", "Fever", "Oxygen_Saturation"
    ])

    return df

# -------------------------
# CSV File Input Function
# -------------------------
def file_input():
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is None:
        return None

    df = pd.read_csv(uploaded_file)

    required_cols = [
        "Age", "Sex", "Comorbidities", "CRP", "WBC", "Fever", "Oxygen_Saturation"
    ]

    # Check which columns exist
    available_cols = [col for col in required_cols if col in df.columns]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.warning(
            f"⚠️ The following columns are missing and will be ignored: {', '.join(missing_cols)}"
        )
    
    if "Sex" in available_cols:
        # Encode Sex column if it exists
        df["Sex"] = sex_encoder.transform(df["Sex"].astype(str))

    # Only keep available columns needed for prediction
    df_valid = df[available_cols].copy()

    return df_valid


# -------------------------
# Prediction Function
# -------------------------
def run_prediction(input_df):
    # Make sure we have all features needed by the model
    model_features = [
        "Age", "Sex", "Comorbidities", "CRP", "WBC", "Fever", "Oxygen_Saturation"
    ]
    for f in model_features:
        if f not in input_df.columns:
            # Fill missing feature with median values for safe prediction
            input_df[f] = 0  # or you can use input_df[f].median() if numeric

    scaled = scaler.transform(input_df[model_features])

    # Severity prediction
    severity_pred = severity_model.predict(scaled)
    severity_prob = severity_model.predict_proba(scaled).max(axis=1)

    # ICU prediction
    icu_pred = icu_model.predict(scaled)
    icu_prob = icu_model.predict_proba(scaled)[:, 1]

    results = input_df.copy()
    results["Severity"] = severity_encoder.inverse_transform(severity_pred)
    results["Severity_Confidence"] = np.round(severity_prob, 2)
    results["ICU_Risk_%"] = np.round(icu_prob * 100, 2)
    results["ICU_Prediction"] = icu_pred

    results["ICU_Risk_Level"] = pd.cut(
        results["ICU_Risk_%"],
        bins=[0, 30, 70, 100],
        labels=["Low", "Medium", "High"]
    )

    return results

# -------------------------
# SHAP Explainability Function
# -------------------------
# -------------------------
# SHAP Explainability Function
# -------------------------
def shap_explain(model, input_df):
    st.subheader("🧠 SHAP Feature Importance")

    try:
        X = input_df.copy()

        # Create SHAP explainer
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Loop through patients
        for i in range(len(X)):
            st.write(f"Patient {i+1}")

            # For multi-class, shap_values[i].values may be 2D
            # Take the mean across classes if needed
            if shap_values[i].values.ndim > 1:
                shap_vals_to_plot = shap_values[i].values.mean(axis=0)
            else:
                shap_vals_to_plot = shap_values[i].values

            fig, ax = plt.subplots()
            shap.bar_plot(shap_vals_to_plot, feature_names=X.columns, max_display=10)
            st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP explainability could not be generated: {e}")


# -------------------------
# Main App Logic
# -------------------------
results = None  # initialize

if input_mode == "Manual Entry":
    input_df = manual_input()
    if st.button("Predict"):
        results = run_prediction(input_df)
        st.subheader("📊 Prediction Results")
        st.dataframe(results)
        shap_explain(severity_model, input_df)

elif input_mode == "Upload CSV File":
    input_df = file_input()
    if input_df is not None and st.button("Run Batch Prediction"):
        results = run_prediction(input_df)
        st.subheader("📊 Batch Prediction Results")
        st.dataframe(results)

        st.download_button(
            label="Download Results CSV",
            data=results.to_csv(index=False),
            file_name="cdss_results.csv"
        )

        shap_explain(severity_model, input_df)

# -------------------------
# Only show charts if results exist
# -------------------------
if results is not None:
    st.subheader("📈 Severity Distribution")
    st.bar_chart(results["Severity"].value_counts())

    st.subheader("📈 ICU Risk Distribution")
    st.bar_chart(results["ICU_Risk_Level"].value_counts())
