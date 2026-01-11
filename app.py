import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.title("Student Exam Score Predictor")

st.write("Enter student details to predict exam score")

# User inputs
study_hours = st.slider("Study Hours per Day", 1, 10, 5)
attendance = st.slider("Attendance Percentage", 50, 100, 75)

# Prepare input
input_data = pd.DataFrame({"study_hours": [study_hours],"attendance": [attendance]})

# Prediction
if st.button("Predict Score"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Exam Score: {round(prediction[0], 2)}")
