import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# LOAD MODEL + FEATURE ORDER
# ---------------------------
model = joblib.load("models/house_price_model.pkl")
model_features = joblib.load("models/model_features.pkl")

# ---------------------------
# APP UI
# ---------------------------
st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏡 House Price Prediction App")
st.write("Enter property details to estimate market value.")

st.write("---")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input("Living Area (sq ft)", 300, 6000, 1500)
    total_bsmt = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
    full_bath = st.slider("Full Bathrooms", 1, 5, 2)
    year_built = st.number_input("Year Built", 1900, 2025, 2005)

with col2:
    overall_qual = st.slider("Overall Construction Quality (1-10)", 1, 10, 5)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
    rooms = st.number_input("Total Rooms Above Ground", 2, 15, 6)

st.write("---")

if st.button("🔍 Estimate Price"):
    
    raw_input = {
    "GrLivArea": gr_liv_area,
    "OverallQual": overall_qual,
    "TotalBsmtSF": total_bsmt,
    "GarageCars": garage_cars,
    "YearBuilt": year_built,
    "FullBath": full_bath,
    "TotRmsAbvGrd": rooms
}


    input_df = pd.DataFrame([raw_input])
    input_df = input_df[model_features]  # ✅ ensures correct order
    
    


    pred_log = model.predict(input_df)[0]
    price = np.expm1(pred_log)   # ✅ convert from log scale
    price = float(price)
  # avoid ridiculous values
    price = round(float(price))

    st.success(f"### 💰 Estimated House Price: **₹ {price:,.0f}**")

    if price < 3000000:
        st.info("🏠 Category: Affordable")
    elif price < 8000000:
        st.warning("🏡 Category: Mid-Range")
    else:
        st.error("🏰 Category: Luxury")

st.write("---")
st.caption("Developed by Sri Varshini, Srushti & Sirisha ✨")
st.caption("Guided by  Prof. Dr.Chaitra (dept. of AIML, RNSIT)")
st.caption("Data Source: Kaggle - House Prices: Advanced Regression Techniques")
st.caption("Model: XGBoost Regressor")

