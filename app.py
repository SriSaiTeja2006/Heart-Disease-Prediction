import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Load model and files (JOBLIB)
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("❤️ Cardiovascular Disease Prediction")

st.markdown(
    """
    Heart disease, also known as cardiovascular disease, is one of the most serious
    illnesses worldwide. This application predicts the risk of heart disease using
    Machine Learning.
    """
)

st.subheader("🧾 Enter Patient Details")

inputs = []
for feature in features:
    value = st.number_input(feature, min_value=0.0, step=1.0)
    inputs.append(value)

if st.button("🔍 Predict"):
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Cardiovascular Disease")
    else:
        st.success("✅ Low Risk of Cardiovascular Disease")

