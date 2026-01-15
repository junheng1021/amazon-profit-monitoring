import streamlit as st
import pandas as pd
import joblib
import time

st.title("Amazon Profit Prediction (Model v1)")

# Load model
model = joblib.load("models/model_v1.pkl")

st.subheader("Input Order Details")

sales = st.number_input("Sales", min_value=0.0, value=100.0)
quantity = st.number_input("Quantity", min_value=1, value=1)

if st.button("Predict Profit"):
    input_df = pd.DataFrame({
        "Sales": [sales],
        "Quantity": [quantity]
    })

    start_time = time.time()
    prediction = model.predict(input_df)[0]
    latency = time.time() - start_time

    st.subheader("Prediction Result")
    st.write(f"Predicted Profit: **{prediction:.2f}**")
    st.write(f"Prediction Latency: **{latency:.4f} seconds**")
