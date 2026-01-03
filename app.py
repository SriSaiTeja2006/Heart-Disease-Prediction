import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("❤️ Heart Disease Prediction")

inputs = []
for feature in features:
    value = st.number_input(feature)
    inputs.append(value)

if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
