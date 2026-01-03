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

# ---------------- DISCLAIMER ----------------
with st.expander("🏥 Medical Disclaimer"):
    st.warning(
        "This application is for educational purposes only and does not replace "
        "professional medical advice. Always consult a healthcare provider."
    )

# ---------------- INPUT SECTION ----------------
st.subheader("🧾 Patient Health Parameters")

col1, col2 = st.columns(2)

# ---------- COLUMN 1 ----------
with col1:
    age = st.slider("Age (Years)", 18, 100, 40)

    gender_text = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_text == "Male" else 0

    height = st.slider("Height (cm)", 120, 220, 170)
    weight = st.slider("Weight (kg)", 30, 200, 70)

    systolic_bp = st.slider(
        "Systolic Blood Pressure (mmHg)", 80, 200, 120
    )

    cholesterol_text = st.selectbox(
        "Cholesterol Level",
        ["Normal", "Above Normal", "Well Above Normal"]
    )
    cholesterol = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }[cholesterol_text]

# ---------- COLUMN 2 ----------
with col2:
    diastolic_bp = st.slider(
        "Diastolic Blood Pressure (mmHg)", 50, 130, 80
    )

    glucose_text = st.selectbox(
        "Glucose Level",
        ["Normal", "Above Normal", "Well Above Normal"]
    )
    glucose = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }[glucose_text]

    smoking_text = st.selectbox("Smoking Habit", ["No", "Yes"])
    smoking = 1 if smoking_text == "Yes" else 0

    alcohol_text = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    alcohol = 1 if alcohol_text == "Yes" else 0

    physical_activity_text = st.selectbox(
        "Physical Activity Level",
        ["Inactive", "Moderately Active", "Highly Active"]
    )
    physical_activity = {
        "Inactive": 0,
        "Moderately Active": 1,
        "Highly Active": 1
    }[physical_activity_text]

# ---------------- BMI CALCULATION ----------------
st.markdown("---")
st.subheader("🧮 Body Mass Index (BMI)")

height_m = height / 100
bmi = round(weight / (height_m ** 2), 2)

bmi_col1, bmi_col2 = st.columns(2)

with bmi_col1:
    st.metric("BMI Value", bmi)

with bmi_col2:
    if bmi < 18.5:
        st.info("Underweight")
    elif 18.5 <= bmi < 25:
        st.success("Normal Weight")
    elif 25 <= bmi < 30:
        st.warning("Overweight")
    else:
        st.error("Obese")

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

    # Prediction
    prediction = model.predict(input_scaled)

    # -------- SAFE RISK PERCENTAGE (SVM FIX) --------
    if hasattr(model, "predict_proba"):
        risk_prob = model.predict_proba(input_scaled)[0][1] * 100
    else:
        decision_score = model.decision_function(input_scaled)[0]
        risk_prob = (1 / (1 + np.exp(-decision_score))) * 100

    st.subheader("🩺 Prediction Result")

    st.metric("📊 Estimated Risk Percentage", f"{risk_prob:.2f} %")

    if prediction[0] == 1:
        st.error(
            "⚠️ **High Risk of Cardiovascular Disease**\n\n"
            "Please consult a healthcare professional immediately."
        )
    else:
        st.success(
            "✅ **Low Risk of Cardiovascular Disease**\n\n"
            "Maintain a healthy lifestyle and regular medical checkups."
        )

    # ---------------- FEATURE EXPLANATION ----------------
    st.markdown("---")
    st.subheader("🧠 Key Factors Influencing Prediction")

    explanations = []

    if age > 50:
        explanations.append("• Higher age increases cardiovascular risk.")
    if systolic_bp > 140 or diastolic_bp > 90:
        explanations.append("• Elevated blood pressure increases heart disease risk.")
    if cholesterol > 1:
        explanations.append("• High cholesterol contributes to cardiovascular disease.")
    if glucose > 1:
        explanations.append("• Elevated glucose levels increase cardiovascular risk.")
    if smoking == 1:
        explanations.append("• Smoking is a major risk factor for heart disease.")
    if alcohol == 1:
        explanations.append("• Alcohol consumption may negatively affect heart health.")
    if physical_activity == 0:
        explanations.append("• Low physical activity increases cardiovascular risk.")
    if bmi >= 30:
        explanations.append("• Obesity is strongly associated with heart disease.")

    if explanations:
        for exp in explanations:
            st.write(exp)
    else:
        st.write("• All major health indicators are within normal ranges.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>Machine Learning Project | Streamlit Medical Dashboard</center>",
    unsafe_allow_html=True
)
