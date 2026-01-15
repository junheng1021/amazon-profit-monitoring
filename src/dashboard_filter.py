import streamlit as st
import pandas as pd

st.title("Amazon Sales Dashboard with Search & Filters")

@st.cache_data
def load_data():
    return pd.read_csv("data/amazon_raw.csv")

df = load_data()

st.sidebar.header("Filter Options")

# --- Optional categorical filters (only if columns exist) ---
if "Category" in df.columns:
    category = st.sidebar.selectbox("Select Category", ["All"] + sorted(df["Category"].dropna().unique().tolist()))
    if category != "All":
        df = df[df["Category"] == category]

if "Geography" in df.columns:
    geography = st.sidebar.selectbox("Select Geography", ["All"] + sorted(df["Geography"].dropna().unique().tolist()))
    if geography != "All":
        df = df[df["Geography"] == geography]

if "Sales" in df.columns:
    # Make sure Sales is numeric
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    min_sales = float(df["Sales"].min())
    max_sales = float(df["Sales"].max())

    # Only show slider when there is a valid range
    if pd.notna(min_sales) and pd.notna(max_sales) and min_sales < max_sales:
        sales_range = st.sidebar.slider(
            "Sales Range",
            min_value=min_sales,
            max_value=max_sales,
            value=(min_sales, max_sales)
        )
        df = df[(df["Sales"] >= sales_range[0]) & (df["Sales"] <= sales_range[1])]
    else:
        st.sidebar.info(f"Sales Range filter disabled (Sales min=max={min_sales:.2f}).")


# --- Search across all columns ---
search_term = st.sidebar.text_input("Search (any field)")
if search_term:
    df = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]

st.subheader("Filtered Results")
st.dataframe(df)

# --- Simple summary metrics ---
st.subheader("Summary Metrics")
if "Sales" in df.columns:
    st.metric("Total Sales (Filtered)", f"{df['Sales'].sum():,.2f}")
if "Profit" in df.columns:
    st.metric("Total Profit (Filtered)", f"{df['Profit'].sum():,.2f}")

# --- Simple chart (Profit by Sales bins) ---
st.subheader("Quick Visualization")
if "Sales" in df.columns and "Profit" in df.columns and len(df) > 0:
    chart_df = df[["Sales", "Profit"]].copy()
    st.scatter_chart(chart_df, x="Sales", y="Profit")
else:
    st.info("Not enough data/columns to show chart.")
