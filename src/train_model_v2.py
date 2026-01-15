import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/amazon_raw.csv")

# Target
y = df["Profit"]

# Features (extended)
numeric_features = ["Sales", "Quantity"]
categorical_features = []

# Add categorical columns if they exist
for col in ["Category", "Geography"]:
    if col in df.columns:
        categorical_features.append(col)

X = df[numeric_features + categorical_features]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Improved model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train
pipeline.fit(X, y)

# Save model v2
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model_v2.pkl")

print("Model v2 trained and saved as models/model_v2.pkl")
