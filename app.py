import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. Load the Trained Pipeline ---
@st.cache_resource # Use cache to load model only once
def load_model():
    return joblib.load('cad_final_model.pkl')

model_pipeline = load_model()

# --- 2. App UI Header ---
st.set_page_config(page_title="CAD Risk Advisor", layout="wide")
st.title("🫀 CAD Clinical Decision Support System")
st.markdown("""
This tool uses a Machine Learning model (Random Forest) to estimate the risk of **Coronary Artery Disease (CAD)** based on clinical metrics. It also provides a **SHAP explanation** to show which factors influenced the prediction.
""")

# --- 3. Sidebar Inputs for Doctors ---
st.sidebar.header("Patient Clinical Data")

def get_user_input():
    # Numeric Inputs
    age = st.sidebar.number_input("Age", 20, 100, 50)
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalch = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    ca = st.sidebar.selectbox("Major Vessels Colored by Flourosopy (ca)", [0.0, 1.0, 2.0, 3.0])

    # Categorical Inputs
    sex = st.sidebar.selectbox("Gender", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [True, False])
    restecg = st.sidebar.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])
    exang = st.sidebar.selectbox("Exercise Induced Angina", [True, False])
    slope = st.sidebar.selectbox("ST Slope Type", ["upsloping", "flat", "downsloping"])
    thal = st.sidebar.selectbox("Thalassemia Status", ["normal", "fixed defect", "reversable defect"])

    # Collect into a Dictionary (Must match original feature names exactly)
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalch': thalch, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# --- 4. Prediction & Display ---
st.subheader("Diagnostic Results")

if st.button("Run Risk Assessment"):
    # Make Prediction
    prediction = model_pipeline.predict(input_df)[0]
    probability = model_pipeline.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### Predicted Probability: **{probability:.2%}**")
        if prediction == 1:
            st.error("Conclusion: HIGH RISK of Coronary Artery Disease")
        else:
            st.success("Conclusion: LOW RISK / Healthy")

    # --- 5. SHAP Explanation Logic ---
    with col2:
        st.write("### AI Diagnostic Rationale")
        
        # Extract components from pipeline
        rf_model = model_pipeline.named_steps['clf']
        preprocessor = model_pipeline.named_steps['prep']
        
        # Transform the single input row
        transformed_input = preprocessor.transform(input_df)
        
        # Get feature names from the preprocessor
        cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out())
        num_names = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
        feature_names = num_names + cat_names

        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(transformed_input)
        
        # Handle binary classification output structure
        if isinstance(shap_values, list):
            sv = shap_values[1] # Class 1
            bv = explainer.expected_value[1]
        else:
            sv = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
            bv = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

        # Create Explanation Object for Waterfall Plot
        exp = shap.Explanation(
            values=sv[0] if len(sv.shape) > 1 else sv,
            base_values=bv,
            data=transformed_input[0],
            feature_names=feature_names
        )

        # Plot Waterfall
        fig, ax = plt.subplots()
        shap.plots.waterfall(exp, show=False)
        st.pyplot(plt.gcf())

st.markdown("---")
st.caption("Disclaimer: This tool is for research purposes and should not replace professional medical advice.")