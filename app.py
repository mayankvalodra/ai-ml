# app.py

import streamlit as st
import numpy as np
import joblib
import pickle

# Load the model using joblib
data = joblib.load("fuel_model.pkl")


model = data['model']
scaler = data['scaler']
pca = data['pca']
feature_names = data['features']

st.title("🚗 Car Fuel Efficiency Estimator (MPG)")

# Create input fields dynamically
input_data = {}
for feature in feature_names:
    if "origin" in feature:
        input_data[feature] = st.selectbox(f"{feature} (1 or 0)", [0, 1])
    else:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)


# Prepare input for prediction
input_df = np.array([list(input_data.values())])
input_scaled = scaler.transform(input_df)
input_pca = pca.transform(input_scaled)

# Predict
if st.button("Predict MPG"):
    prediction = model.predict(input_pca)
    st.success(f"Estimated Fuel Efficiency: {prediction[0]:.2f} MPG")
