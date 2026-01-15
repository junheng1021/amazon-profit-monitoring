import os
import joblib
import pandas as pd

DATA_PATH = "data/amazon_raw.csv"
MODEL_V1_PATH = "models/model_v1.pkl"
MODEL_V2_PATH = "models/model_v2.pkl"


def test_data_file_exists():
    assert os.path.exists(DATA_PATH), "Dataset file not found at data/amazon_raw.csv"


def test_model_v1_exists():
    assert os.path.exists(MODEL_V1_PATH), "model_v1.pkl not found in models/"


def test_model_v2_exists():
    assert os.path.exists(MODEL_V2_PATH), "model_v2.pkl not found in models/"


def test_model_v1_predicts():
    model = joblib.load(MODEL_V1_PATH)
    # v1 expects Sales and Quantity
    X = pd.DataFrame({"Sales": [100.0], "Quantity": [2]})
    y_pred = model.predict(X)
    assert len(y_pred) == 1


def test_model_v2_predicts():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_V2_PATH)

    # Build one-row input, adding optional columns if present
    row = {"Sales": 100.0, "Quantity": 2}

    if "Category" in df.columns:
        row["Category"] = df["Category"].dropna().iloc[0]

    if "Geography" in df.columns:
        row["Geography"] = df["Geography"].dropna().iloc[0]

    X = pd.DataFrame([row])
    y_pred = model.predict(X)
    assert len(y_pred) == 1
