import streamlit as st
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page layout and title
st.set_page_config(page_title="CAD Risk Predictor", layout="wide")

st.title("🩺 Heart Disease Prediction System (CAD)")
st.write("This application uses a KNN model optimized for clinical screening.")

# 1. Load the pre-trained KNN Pipeline
@st.cache_resource
def load_model():
    # Ensure you have run main.py first to generate this file
    return joblib.load('heart_disease_knn_model.pkl')

try:
    model = load_model()
except:
    st.error("Model file not found! Please run your training script (main.py) first.")

# 2. Sidebar for Patient Feature Input
st.sidebar.header("Patient Clinical Features")

def get_user_input():
    # Numerical Features
    age = st.sidebar.slider('Age', 1, 100, 50)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 600, 240)
    thalch = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', 0.0, 6.0, 1.0)
    
    # Categorical Features
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type', ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'))
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('True', 'False'))
    restecg = st.sidebar.selectbox('Resting ECG Results', ('normal', 'st-t abnormality', 'lv hypertrophy'))
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('Yes', 'No'))
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST', ('upsloping', 'flat', 'downsloping'))
    thal = st.sidebar.selectbox('Thalassemia', ('normal', 'fixed defect', 'reversable defect'))

    # Construct the feature dictionary
    data = {
        'age': age, 'sex': sex.lower(), 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs.lower() == 'true', 'restecg': restecg,
        'thalch': thalch, 'exang': exang.lower() == 'yes',
        'oldpeak': oldpeak, 'slope': slope, 'thal': thal
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# 3. Sensitivity Selection (Threshold Moving)
st.sidebar.markdown("---")
st.sidebar.subheader("Safety Settings")
# Lowering the threshold increases Recall (fewer missed cases)
threshold = st.sidebar.slider('Sensitivity Threshold (Lower = Safer)', 0.1, 0.9, 0.5, 0.05)

# 4. Main Panel Display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Profile Summary")
    # Convert all values to strings to prevent Arrow conversion errors
    # This ensures "male" and "50" can exist in the same display table
    display_df = input_df.T.astype(str).rename(columns={0: 'Value'})
    st.table(display_df) 

with col2:
    st.subheader("Prediction Analysis")
    # Use a unique key for the button to avoid session state issues
    if st.button("Run Diagnostic Prediction", key="predict_btn"):
        # Get probabilities for the positive class (CAD)
        prob = model.predict_proba(input_df)[:, 1][0]
        
        # Apply custom threshold for "Better safe than sorry" approach
        # A person is classified as 1 (High Risk) if their probability >= threshold
        prediction = 1 if prob >= threshold else 0
        
        if prediction == 1:
            st.error(f"### Result: HIGH RISK")
            st.write(f"The model detected potential CAD indicators.")
        else:
            st.success(f"### Result: LOW RISK")
            st.write(f"The patient appears to be in a low-risk category.")
            
        st.metric(label="CAD Probability Score", value=f"{prob:.2%}")
        
        # Display safety warning if the sensitivity is set high
        if threshold < 0.5:
            st.warning(f"Note: Sensitivity is currently set to {threshold}. "
                       f"The model is biased towards detecting illness to minimize missing cases.")

# Footer with medical disclaimer
st.info("**Disclaimer:** This is a research tool based on the UCI Heart Disease Dataset. "
        "Results must be verified by a qualified cardiologist.")