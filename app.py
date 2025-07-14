import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model_lightgbm_v2.pkl")

st.title("💼 Salary Prediction App")

# Collect inputs based on actual training features
age = st.slider("Age", 18, 90, 30)
education_num = st.slider("Education Level (1–16)", 1, 16, 9)
hours_per_week = st.slider("Hours per week", 1, 99, 40)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
sex = st.selectbox("Sex", ["Female", "Male"])
race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
occupation = st.selectbox("Occupation (encoded)", list(range(15)))  # Replace if you have labels

# Encoding
sex_encoded = 1 if sex == "Male" else 0
race_encoded = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"].index(race)

# Build input in exact order and shape
input_data = np.array([[age, education_num, hours_per_week, capital_gain, capital_loss, sex_encoded, race_encoded, occupation]])

if st.button("Predict Income"):
    prediction = model.predict(input_data)
    result = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"💰 Predicted Income: **{result}**")

    # Optional: estimated salary
    fig, ax = plt.subplots()
    ax.bar(["Estimated Salary"], [75000 if prediction[0] == 1 else 30000], color="#1f77b4")
    ax.set_ylabel("USD")
    st.pyplot(fig)
