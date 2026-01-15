import streamlit as st
import pandas as pd
import joblib
import time

from log_utils import log_prediction  # keep this import style inside src/

st.title("Amazon Profit Prediction (Model v1 vs v2)")

# Load models
model_v1 = joblib.load("models/model_v1.pkl")
model_v2 = joblib.load("models/model_v2.pkl")

# Load dataset (for dropdown values, optional)
@st.cache_data
def load_data():
    return pd.read_csv("data/amazon_raw.csv")

df_raw = load_data()

st.subheader("Input Order Details")
sales = st.number_input("Sales", min_value=0.0, value=100.0)
quantity = st.number_input("Quantity", min_value=1, value=1)

# Optional categorical inputs for v2 (only if columns exist)
category = None
geography = None

if "Category" in df_raw.columns:
    category = st.selectbox("Category", sorted(df_raw["Category"].dropna().unique().tolist()))
else:
    st.info("Category column not found in dataset; v2 will run without Category.")

if "Geography" in df_raw.columns:
    geography = st.selectbox("Geography", sorted(df_raw["Geography"].dropna().unique().tolist()))
else:
    st.info("Geography column not found in dataset; v2 will run without Geography.")

st.subheader("User Feedback")
feedback_score = st.slider("Feedback Score (1 = poor, 5 = excellent)", 1, 5, 3)
feedback_comment = st.text_area("Comments (optional)")

if st.button("Predict Profit (Compare v1 vs v2)"):
    # ----- v1 input -----
    input_v1 = pd.DataFrame({
        "Sales": [sales],
        "Quantity": [quantity]
    })

    # ----- v2 input (uses extra columns if available) -----
    input_v2_dict = {
        "Sales": [sales],
        "Quantity": [quantity]
    }
    if category is not None:
        input_v2_dict["Category"] = [category]
    if geography is not None:
        input_v2_dict["Geography"] = [geography]

    input_v2 = pd.DataFrame(input_v2_dict)

    # Predict v1 + latency
    t1 = time.time()
    pred_v1 = float(model_v1.predict(input_v1)[0])
    latency_v1 = time.time() - t1

    # Predict v2 + latency
    t2 = time.time()
    pred_v2 = float(model_v2.predict(input_v2)[0])
    latency_v2 = time.time() - t2

    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model v1 (Baseline)")
        st.write(f"Predicted Profit: **{pred_v1:.2f}**")
        st.write(f"Latency: **{latency_v1:.4f} s**")

    with col2:
        st.markdown("### Model v2 (Improved)")
        st.write(f"Predicted Profit: **{pred_v2:.2f}**")
        st.write(f"Latency: **{latency_v2:.4f} s**")

    # Log both predictions (two rows)
    log_prediction(
        model_version="v1",
        sales=float(sales),
        quantity=int(quantity),
        prediction=pred_v1,
        latency_seconds=float(latency_v1),
        feedback_score=int(feedback_score),
        feedback_comment=str(feedback_comment)
    )

    log_prediction(
        model_version="v2",
        sales=float(sales),
        quantity=int(quantity),
        prediction=pred_v2,
        latency_seconds=float(latency_v2),
        feedback_score=int(feedback_score),
        feedback_comment=str(feedback_comment)
    )

    st.success("Saved v1 and v2 results to data/monitoring_logs.csv")
