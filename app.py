import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Set page layout and title
st.set_page_config(page_title="CAD Risk Predictor", layout="wide")

st.title("🩺 Heart Disease Prediction System (CAD)")
st.write("This application uses a KNN model optimized for clinical screening.")

# 1. Load the pre-trained KNN Pipeline and SHAP Metadata
@st.cache_resource
def load_assets():
    model = joblib.load('heart_disease_knn_model.pkl')
    shap_meta = joblib.load('shap_metadata.pkl')
    return model, shap_meta

try:
    model, shap_meta = load_assets()
except Exception as e:
    st.error(f"Required files not found! Error: {e}")

# 2. Sidebar for Patient Feature Input
st.sidebar.header("Patient Clinical Features")

def get_user_input():
    age = st.sidebar.slider('Age', 1, 100, 50)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 600, 240)
    thalch = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', 0.0, 6.0, 1.0)
    
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type', ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'))
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('True', 'False'))
    restecg = st.sidebar.selectbox('Resting ECG Results', ('normal', 'st-t abnormality', 'lv hypertrophy'))
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('Yes', 'No'))
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST', ('upsloping', 'flat', 'downsloping'))
    thal = st.sidebar.selectbox('Thalassemia', ('normal', 'fixed defect', 'reversable defect'))

    data = {
        'age': age, 'sex': sex.lower(), 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs.lower() == 'true', 'restecg': restecg,
        'thalch': thalch, 'exang': exang.lower() == 'yes',
        'oldpeak': oldpeak, 'slope': slope, 'thal': thal
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# 3. Sensitivity Selection
st.sidebar.markdown("---")
st.sidebar.subheader("Safety Settings")
threshold = st.sidebar.slider('Sensitivity Threshold (Lower = Safer)', 0.1, 0.9, 0.5, 0.05)

# 4. Main Panel Display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Profile Summary")
    display_df = input_df.T.astype(str).rename(columns={0: 'Value'})
    st.table(display_df) 

with col2:
    st.subheader("Prediction Analysis")
    if st.button("Run Diagnostic Prediction", key="predict_btn"):
        # Prediction
        prob = model.predict_proba(input_df)[:, 1][0]
        prediction = 1 if prob >= threshold else 0
        
        if prediction == 1:
            st.error(f"### Result: HIGH RISK")
        else:
            st.success(f"### Result: LOW RISK")
        st.metric(label="CAD Probability Score", value=f"{prob:.2%}")

        # --- SHAP EXPLAINER LOGIC ---
        st.markdown("---")
        st.subheader("💡 Diagnostic Explanation (XAI)")
        
        plot_placeholder = st.empty()
        
        with st.spinner("Analyzing clinical contribution..."):
            try:
                # 1. Preprocess the current input
                input_transformed = model.named_steps['features'].transform(input_df)
                if hasattr(input_transformed, "toarray"):
                    input_transformed = input_transformed.toarray()

                # 2. Get the model from the pipeline (last step)
                knn_model = model.steps[-1][1]

                # 3. Create Explainer
                explainer = shap.KernelExplainer(knn_model.predict_proba, shap_meta['background_data'])
                
                # 4. Compute SHAP Values (limiting nsamples for speed)
                shap_values = explainer.shap_values(input_transformed, nsamples=100)
                
                # 5. Handle indexing for Waterfall Plot (CRITICAL FIX)
                # For classification, we want the results for the Disease class (index 1)
                # s_vals should be a 1D array of shape (features,)
                if isinstance(shap_values, list):
                    # KernelExplainer typically returns [class0_array, class1_array]
                    s_vals = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    # If it's a single array of shape (samples, features, classes)
                    s_vals = shap_values[0, :, 1]
                    base_val = explainer.expected_value[1]

                # 6. Build the single-sample Explanation object
                exp = shap.Explanation(
                    values=s_vals,
                    base_values=base_val,
                    data=input_transformed[0],
                    feature_names=shap_meta['feature_names']
                )

                # 7. Rendering
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(exp, show=False)
                plt.tight_layout()
                
                plot_placeholder.pyplot(fig)
                st.info("The Waterfall chart explains which features pushed the risk up (red) or down (blue).")
                
            except Exception as e:
                st.error(f"SHAP Error: {e}")

# Footer
st.info("**Disclaimer:** Results must be verified by a qualified cardiologist.")