# ❤️ Heart Disease Prediction System

This project is a Machine Learning–based web application that predicts the risk of cardiovascular (heart) disease based on patient health parameters.  
The model is trained using multiple ML algorithms, and the best-performing model is deployed using **Streamlit**.

---

## 📊 Project Overview

Cardiovascular diseases are one of the leading causes of death worldwide.  
This project aims to assist in early detection of heart disease by analyzing medical attributes using Machine Learning techniques.

---

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  
- Google Colab (Model Training)  
- GitHub & Streamlit Cloud (Deployment)

---

## 📁 Dataset

- Cardiovascular Disease Dataset (`cardio_train.csv`)
- Features include:
  - Age
  - Gender
  - Height
  - Weight
  - Blood Pressure
  - Cholesterol
  - Glucose
  - Smoking, Alcohol intake
  - Physical activity

---

## ⚙️ Project Workflow

1. Data Pre-processing  
2. Exploratory Data Analysis (EDA) & Visualizations  
3. Correlation Matrix Analysis  
4. Model Training and Evaluation  
5. Accuracy Comparison of Models  
6. Model Selection  
7. Deployment using Streamlit  

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  

### ✅ Best Model
- **Support Vector Machine (SVM)** achieved the highest accuracy and was selected for deployment.

---

## 📈 Model Accuracy Comparison

| Model | Accuracy |
|------|---------|
| Logistic Regression | ~72% |
| KNN | ~64% |
| **SVM (Selected)** | **~72.6%** |
| Decision Tree | ~63% |
| Random Forest | ~71% |

---

## 🌐 Web Application (Streamlit)

The trained model is deployed as a web application using **Streamlit**, where users can input medical details and receive a prediction indicating:

- ✅ Low Risk of Heart Disease  
- ⚠️ High Risk of Heart Disease  

---

## 🚀 Deployment

- Platform: **Streamlit Community Cloud**
- Repository contains:
  - `app.py`
  - `requirements.txt`
  - Trained model files (`.pkl`)

