import streamlit as st
import pandas as pd

st.title("Amazon Sales Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("data/amazon_raw.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Summary Metrics")

if "Sales" in df.columns:
    st.metric("Total Sales", f"{df['Sales'].sum():,.2f}")

if "Profit" in df.columns:
    st.metric("Total Profit", f"{df['Profit'].sum():,.2f}")
