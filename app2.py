import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("insurance_model.pkl", "rb"))

st.title("Insurance Charge Prediction")

age = st.slider("Age", 18, 100)
bmi = st.slider("BMI", 10.0, 50.0)
children = st.slider("Children", 0, 10)
smoker = st.checkbox("Smoker")
sex = st.radio("Sex", ("Male", "Female"))

if st.button("Predict"):

    features = np.array([
        age,
        bmi,
        children,
        1 if smoker else 0,
        1 if sex == "Male" else 0
    ]).reshape(1, -1)

    prediction = model.predict(features)

    st.success(f"Predicted Insurance Charge: ₹{prediction[0]:,.2f}")