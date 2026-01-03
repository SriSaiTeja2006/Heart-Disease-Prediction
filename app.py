import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='color:#d6336c;'>❤️ Cardiovascular Disease Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "AI-powered system to assess heart disease risk using patient health parameters."
)
st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("🧾 Patient Health Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 18, 100, 40)

    gender_text = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_text == "Male" else 0

    height = st.slider("Height (cm)", 120, 220, 170)
    weight = st.slider("Weight (kg)", 30, 200, 70)

    systolic_bp = st.slider(
        "Systolic Blood Pressure (mmHg)", 80, 200, 120
    )

    cholesterol = st.selectbox(
        "Cholesterol Level",
        ["Normal", "Above Normal", "Well Above Normal"]
    )
    cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]

with col2:
    diastolic_bp = st.slider(
        "Diastolic Blood Pressure (mmHg)", 50, 130, 80
    )

    glucose = st.selectbox(
        "Glucose Level",
        ["Normal", "Above Normal", "Well Above Normal"]
    )
    glucose = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[glucose]

    smoking = st.slider("Smoking Habit (%)", 0, 100, 0)
    smoking = 1 if smoking > 0 else 0

    alcohol = st.slider("Alcohol Consumption (%)", 0, 100, 0)
    alcohol = 1 if alcohol > 0 else 0

    physical_activity = st.selectbox(
        "Physical Activity Level",
        ["Inactive", "Moderately Active", "Highly Active"]
    )
    physical_activity = {
        "Inactive": 0,
        "Moderately Active": 1,
        "Highly Active": 1
    }[physical_activity]

# ---------------- PREDICTION ----------------
st.markdown("---")

if st.button("🔍 Predict Cardiovascular Risk", use_container_width=True):

    input_data = np.array([[
        age,
        gender,
        height,
        weight,
        systolic_bp,
        diastolic_bp,
        cholesterol,
        glucose,
        smoking,
        alcohol,
        physical_activity
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.subheader("🩺 Prediction Result")

    if prediction[0] == 1:
        st.error(
            "⚠️ **High Risk of Cardiovascular Disease**\n\n"
            "Please consult a healthcare professional for further evaluation."
        )
    else:
        st.success(
            "✅ **Low Risk of Cardiovascular Disease**\n\n"
            "Maintain a healthy lifestyle and regular medical checkups."
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>Machine Learning Project | Streamlit Medical Dashboard</center>",
    unsafe_allow_html=True
)

