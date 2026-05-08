# streamlit_app.py — Extended Edition
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

#Page Configuration
st.set_page_config(
    page_title="Malaria Predictor",
    page_icon="\U0001F99F",
    layout="centered"
)

st.title("\U0001F99F Malaria Severity Predictor")
st.divider()

# -------------------------------------------------
# 1. Load the model & feature list once
# -------------------------------------------------
@st.cache_resource # load once, reuse on every rerun

def load_model():
    return (joblib.load('model/pipeline.joblib'),
            joblib.load('model/features.joblib'))

pipeline, FEATURES = load_model()

# Grab the underlying estimator for feature-importance & SHAP use
model = (pipeline.named_steps.get('model')
         or pipeline.named_steps.get('classifier')
         or pipeline.named_steps.get('clf')
         or pipeline.steps[-1][1])

# -------------------------------------------------
# 2. SIDEBAR: Top-5 Feature Chart + Request Counter
# -------------------------------------------------
st.sidebar.header("\U0001F4CA Model Insights")

# --- Top 5 feature importances (bar chart) ---
if model is not None and hasattr(model, 'feature_importances_'):
    importances = pd.Series(
        model.feature_importances_,
        index=FEATURES,
        name="Importance"
    ).sort_values(ascending=False).head(5)
    st.sidebar.subheader("Top 5 Important Features")
    st.sidebar.bar_chart(importances)
else:
    st.sidebar.info("\U000126A0 Model does not expose feature_importances_.")

# --- Session-state prediction counter ---
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

st.sidebar.subheader("\U0001F522 Request Counter")
st.sidebar.metric("Predictions made", st.session_state.prediction_count)

# -------------------------------------------------
# 3. MAIN PAGE — Single Patient Prediction
# -------------------------------------------------
age = st.number_input('Age (years)', min_value=0, max_value=120, value=25)
sex = 1 if st.selectbox('Sex', ['Female', 'Male']) == 'Male' else 0

st.markdown("### Symptoms (Check all that apply)")
# Symptom Checkboxes
col1, col2, col3 = st.columns(3)
with col1:
    fever = 1 if st.checkbox('Fever') else 0
    cold = 1 if st.checkbox('Cold') else 0
    rigor = 1 if st.checkbox('Rigor') else 0
    fatigue = 1 if st.checkbox('Fatigue') else 0
    headache = 1 if st.checkbox('Headache') else 0
with col2:
    bitter_tongue = 1 if st.checkbox('Bitter Tongue') else 0
    vomitting = 1 if st.checkbox('Vomiting') else 0
    diarrhea = 1 if st.checkbox('Diarrhea') else 0
    convulsion = 1 if st.checkbox('Convulsion') else 0
    anemia = 1 if st.checkbox('Anemia') else 0
with col3:
    jaundice = 1 if st.checkbox('Jaundice') else 0
    cocacola_urine = 1 if st.checkbox('Coca-Cola Urine') else 0
    hypoglycemia = 1 if st.checkbox('Hypoglycemia') else 0
    prostration = 1 if st.checkbox('Prostration') else 0
    hyperpyrexia = 1 if st.checkbox('Hyperpyrexia') else 0

if st.button("✓ Run Prediction", type="primary"):
    row = {
        'age': age, 'sex': sex, 'fever': fever, 'cold': cold,
        'rigor': rigor, 'fatigue': fatigue, 'headache': headache,
        'bitter_tongue': bitter_tongue, 'vomitting': vomitting,
        'diarrhea': diarrhea, 'convulsion': convulsion, 'anemia': anemia,
        'jaundice': jaundice, 'cocacola_urine': cocacola_urine,
        'hypoglycemia': hypoglycemia, 'prostration': prostration,
        'hyperpyrexia': hyperpyrexia
    }

    X = pd.DataFrame([row], columns=FEATURES)
    pred = int(pipeline.predict(X)[0])
    prob = float(pipeline.predict_proba(X)[0][1])
    risk = 'HIGH' if prob >= 0.7 else 'MEDIUM' if prob >= 0.4 else 'LOW'

# Update counter (persists for the session)
    st.session_state.prediction_count += 1

# Prediction result
    st.success(f'Prediction: {"Severe" if pred == 1 else "Not Severe"}')
    st.metric('Probability of Severe Malaria', f'{prob:.1%}')

# --- Colour-coded risk badge (HTML) ---
    risk_color = {"LOW": "#28a745", "MEDIUM": "#fd7e14", "HIGH": "#dc3545"}
    st.markdown(
        f'<div style="display:inline-block; padding:6px 14px; border-radius:8px; '
        f'background-color:{risk_color[risk]}; color:white; font-weight:bold; font-size:1.1rem;">'
        f'Risk Level: {risk}</div>',
        unsafe_allow_html=True
    )


# SHAP Explanation Panel
    st.divider()
    st.subheader('Explanation of Prediction (SHAP)')
    try:
        # Wrap the predict pipeline in a SHAP explainer
        # Using the agnostic Explainer (handles pipelines natively by passing the predict function)
        explainer = shap.Explainer(pipeline.predict, X)
        shap_values = explainer(X)
        shap_values.feature_names = FEATURES
        
        # Plotting the waterfall
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation. Note: Ensure your pipeline structure is fully compatible with SHAP. Details: {e}")


# -------------------------------------------------
# 4. BATCH PREDICTION (CSV Upload)
# -------------------------------------------------
st.divider()
with st.expander("\U0001F4C1 Batch Prediction (CSV Upload)", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload a CSV File to run multiple predictions at once. Ensure columns match the features above (exact names, no extra columns).",
        type=["csv"]
    )

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)

        missing_cols = [c for c in FEATURES if c not in df_batch.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            preds = pipeline.predict(df_batch[FEATURES])
            probs = pipeline.predict_proba(df_batch[FEATURES])[:, 1]

            df_batch['Prediction'] = np.where(preds == 1, 'Severe', 'Not Severe')
            df_batch['Probability'] = probs
            df_batch['Risk'] = np.where(
                probs >= 0.7, 'HIGH',
                np.where(probs >= 0.4, 'MEDIUM', 'LOW')
            )

            st.success(f"✅ Batch prediction complete for **{len(df_batch)}** rows.")
            st.dataframe(df_batch.head(50))

            csv_out = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_out,
                file_name="malaria_batch_predictions.csv",
                mime="text/csv"
            )