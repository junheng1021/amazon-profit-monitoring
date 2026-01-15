import streamlit as st
import pandas as pd
import os

LOG_PATH = "data/monitoring_logs.csv"

st.title("Model Monitoring Dashboard")

if not os.path.exists(LOG_PATH):
    st.error("No monitoring logs found yet. Run the prediction app and generate logs first.")
    st.stop()

# Load logs
df = pd.read_csv(LOG_PATH)

# Handle empty logs safely
if df.empty:
    st.warning("monitoring_logs.csv exists but has no rows yet. Generate at least 1 prediction log.")
    st.stop()

st.subheader("Raw Monitoring Logs")
st.dataframe(df)

# Convert types safely
if "latency_seconds" in df.columns:
    df["latency_seconds"] = pd.to_numeric(df["latency_seconds"], errors="coerce")
if "feedback_score" in df.columns:
    df["feedback_score"] = pd.to_numeric(df["feedback_score"], errors="coerce")

st.subheader("Summary Metrics")

col1, col2, col3 = st.columns(3)

# Total predictions
col1.metric("Total Predictions", int(len(df)))

# Average latency
if "latency_seconds" in df.columns:
    avg_latency = df["latency_seconds"].dropna().mean()
    col2.metric("Avg Latency (s)", f"{avg_latency:.4f}" if pd.notna(avg_latency) else "N/A")
else:
    col2.metric("Avg Latency (s)", "N/A")

# Average feedback
if "feedback_score" in df.columns:
    avg_feedback = df["feedback_score"].dropna().mean()
    col3.metric("Avg Feedback Score", f"{avg_feedback:.2f}" if pd.notna(avg_feedback) else "N/A")
else:
    col3.metric("Avg Feedback Score", "N/A")

st.subheader("Model Version Comparison")

# Compare latency and feedback by model_version (works even if only v1 exists)
if "model_version" in df.columns:
    group = df.groupby("model_version", dropna=False).agg(
        avg_latency=("latency_seconds", "mean"),
        avg_feedback=("feedback_score", "mean"),
        count=("model_version", "count")
    ).reset_index()

    st.dataframe(group)

    # Simple visualization
    st.caption("Average latency by model version")
    if "avg_latency" in group.columns and group["avg_latency"].notna().any():
        st.bar_chart(group.set_index("model_version")["avg_latency"])

    st.caption("Average feedback score by model version")
    if "avg_feedback" in group.columns and group["avg_feedback"].notna().any():
        st.bar_chart(group.set_index("model_version")["avg_feedback"])
else:
    st.info("model_version column not found in logs. Please check your logging fields.")

st.subheader("Recent User Comments")

if "feedback_comment" in df.columns:
    recent_comments = df[df["feedback_comment"].astype(str).str.strip() != ""].tail(10)
    if recent_comments.empty:
        st.write("No comments yet.")
    else:
        for _, row in recent_comments.iterrows():
            st.write(f"- **{row.get('timestamp', '')}** ({row.get('model_version','')}): {row.get('feedback_comment','')}")
else:
    st.write("No feedback_comment column found.")
