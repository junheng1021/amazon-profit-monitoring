import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/amazon_raw.csv")

# Check required columns
required_cols = {"Sales", "Quantity", "Profit"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Features and target
X = df[["Sales", "Quantity"]]
y = df["Profit"]

# Train baseline model
model = LinearRegression()
model.fit(X, y)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/model_v1.pkl")

print("Model v1 saved to models/model_v1.pkl")
