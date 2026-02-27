import streamlit as st
import numpy as np
import joblib

# Load Saved Model
model = joblib.load("best_model.pkl")

# Page Title
st.title("🩺 Diabetes Predictor")

st.write(
    "This model predicts diabetes progression using regression."
)

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.write("Best Performing Model: Lasso Regression")
st.sidebar.write("Type: Regression")

# Input Section
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age")
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    bmi = st.number_input("BMI")
    bp = st.number_input("Blood Pressure")
    s1 = st.number_input("s1")

with col2:
    s2 = st.number_input("s2")
    s3 = st.number_input("s3")
    s4 = st.number_input("s4")
    s5 = st.number_input("s5")
    s6 = st.number_input("s6")

# Prediction
if st.button("Predict"):

    if all(v == 0 for v in [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]):
        st.warning("Please enter patient details before predicting.")

    else:
        input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
        prediction = model.predict(input_data)
        st.success(f"Predicted Diabetes Progression Score: {prediction[0]:.2f}")