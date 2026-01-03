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
features = joblib.load("features.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.title {
    font-size: 38px;
    font-weight: 700;
    color: #d6336c;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.result-high {
    background-color: #ffe3e3;
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    color: #c92a2a;
    font-weight: 600;
}
.result-low {
    background-color: #d3f9d8;
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    color: #2b8a3e;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>❤️ Cardiovascular Disease Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered system to assess heart disease risk using medical parameters</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- INFO CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='card'><h4>🌍 Global Impact</h4><p>Heart disease accounts for <b>28.1%</b> of global deaths.</p></div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><h4>🧠 ML Based</h4><p>Prediction using trained Machine Learning models.</p></div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'><h4>⚕️ Early Detection</h4><p>Supports timely diagnosis & prevention.</p></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown("## 🧾 Patient Health Parameters")

col1, col2 = st.columns(2)
inputs = []

for i, feature in enumerate(features):
    if i % 2 == 0:
        with col1:
            value = st.number_input(
                feature.upper(),
                min_value=0.0,
                step=1.0
            )
    else:
        with col2:
            value = st.number_input(
                feature.upper(),
                min_value=0.0,
                step=1.0
            )
    inputs.append(value)

# ---------------- PREDICTION ----------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Cardiovascular Risk", use_container_width=True):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    st.markdown("## 🩺 Prediction Result")

    if prediction[0] == 1:
        st.markdown(
            "<div class='result-high'>⚠️ HIGH RISK of Cardiovascular Disease<br>"
            "Consult a medical professional immediately.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-low'>✅ LOW RISK of Cardiovascular Disease<br>"
            "Maintain a healthy lifestyle & regular checkups.</div>",
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<center>Machine Learning Project | Streamlit Deployment</center>",
    unsafe_allow_html=True
)

