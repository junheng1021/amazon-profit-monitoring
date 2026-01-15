import streamlit as st
import pandas as pd
import joblib
import time

from log_utils import log_prediction


st.title("Amazon Profit Prediction (Model v1)")

# Load model v1
model = joblib.load("models/model_v1.pkl")

st.subheader("Input Order Details")
sales = st.number_input("Sales", min_value=0.0, value=100.0)
quantity = st.number_input("Quantity", min_value=1, value=1)

st.subheader("User Feedback")
feedback_score = st.slider("Feedback Score (1 = poor, 5 = excellent)", 1, 5, 3)
feedback_comment = st.text_area("Comments (optional)")

if st.button("Predict Profit"):
    input_df = pd.DataFrame({"Sales": [sales], "Quantity": [quantity]})

    start_time = time.time()
    prediction = float(model.predict(input_df)[0])
    latency = time.time() - start_time

    st.subheader("Prediction Result")
    st.write(f"Predicted Profit: **{prediction:.2f}**")
    st.write(f"Prediction Latency: **{latency:.4f} seconds**")

    # Log to CSV
    log_prediction(
        model_version="v1",
        sales=float(sales),
        quantity=int(quantity),
        prediction=prediction,
        latency_seconds=float(latency),
        feedback_score=int(feedback_score),
        feedback_comment=str(feedback_comment)
    )

    st.success("Saved to data/monitoring_logs.csv")
