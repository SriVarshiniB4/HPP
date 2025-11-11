import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("models/house_price_model.pkl")

st.title("🏡 House Price Prediction App")

# User inputs
GrLivArea = st.number_input("Above Ground Living Area (sq ft):", min_value=300, max_value=5000, value=1500)
OverallQual = st.slider("Overall Quality (1 - 10):", 1, 10, 5)
GarageCars = st.slider("Number of Garage Cars:", 0, 5, 2)
TotalBsmtSF = st.number_input("Basement Area (sq ft):", min_value=0, max_value=3000, value=800)

# Prepare data for model
input_data = pd.DataFrame([{
    "GrLivArea": GrLivArea,
    "OverallQual": OverallQual,
    "GarageCars": GarageCars,
    "TotalBsmtSF": TotalBsmtSF
}])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: **${prediction:,.2f}**")
