import streamlit as st
import pandas as pd
import joblib

# Load the saved model
pipeline = joblib.load('pipeline_model.pkl')

# Streamlit title
st.title("Purchase Prediction Model")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Salary", min_value=10000, max_value=200000, value=50000)
gender = st.selectbox("Gender", options=["male", "female"])

# Create a DataFrame from the user input
new_data = pd.DataFrame({
    'age': [age],
    'salary': [salary],
    'gender': [gender]
})

# Make a prediction
if st.button("Make Prediction"):
    prediction = pipeline.predict(new_data)
    prediction_proba = pipeline.predict_proba(new_data)

    # Display the results
    st.write(f"Predicted class: {prediction[0]}")
    st.write(f"Probability of class 0: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of class 1: {prediction_proba[0][1]:.2f}")
