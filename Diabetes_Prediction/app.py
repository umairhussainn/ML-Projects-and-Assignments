import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Diabetes Risk Checker", page_icon="💉")

st.title("💉 Diabetes Risk Checker")
st.write("Enter patient details to predict diabetes risk.")

# Paths
model_path = os.path.join("models", "diabetes_xgb.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

# Load model safely
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("❌ Model or scaler not found. Please run training notebook first.")
    st.stop()

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Input UI
left, right = st.columns(2)

with left:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with right:
    insulin = st.number_input("Insulin", 0, 1000, 85)
    bmi = st.number_input("BMI", 10.0, 60.0, 28.5)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.45)
    age = st.number_input("Age", 10, 120, 33)

# Prediction
if st.button("🔍 Check Risk"):
    try:
        sample = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

        sample_scaled = scaler.transform(sample)

        # ✅ Always predict on scaled data
        pred = model.predict(sample_scaled)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(sample_scaled)[0][1]
        else:
            prob = 0.5

        if pred == 1:
            st.error(f"⚠️ High risk of diabetes — Probability: {prob:.2%}")
        else:
            st.success(f"✅ Low risk — Probability: {prob:.2%}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")