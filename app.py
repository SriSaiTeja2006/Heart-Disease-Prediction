import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------
# Load Model & Files
# -------------------------------
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# -------------------------------
# Title Section
# -------------------------------
st.markdown(
    "<h1 style='text-align: center; color: #d6336c;'>❤️ Cardiovascular Disease Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# Project Description
# -------------------------------
st.markdown(
    """
    ### 📌 Project Overview

    Heart disease, also known as **cardiovascular disease**, is one of the most serious
    illnesses in India and across the globe.

    - Cardiac illnesses account for **28.1% of total deaths**
    - More than **17.6 million deaths** worldwide were caused by heart disease
    - Early and accurate prediction is crucial for **timely diagnosis and treatment**

    This system uses **Machine Learning techniques** to analyze patient health data
    and predict the **risk of cardiovascular disease** with reliability and precision.
    """
)

st.markdown("---")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("🧾 Enter Patient Details")

user_inputs = []

for feature in features:
    value = st.number_input(
        label=f"{feature}",
        min_value=0.0,
        step=1.0
    )
    user_inputs.append(value)

# -------------------------------
# Prediction Section
# -------------------------------
st.markdown("---")

if st.button("🔍 Predict Heart Disease Risk"):
    input_array = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    st.markdown("### 🩺 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Cardiovascular Disease**")
        st.markdown(
            "👉 Please consult a medical professional for further diagnosis."
        )
    else:
        st.success("✅ **Low Risk of Cardiovascular Disease**")
        st.markdown(
            "👉 Maintain a healthy lifestyle and regular health checkups."
        )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>"
    "Machine Learning Project | Streamlit Deployment</p>",
    unsafe_allow_html=True
)
