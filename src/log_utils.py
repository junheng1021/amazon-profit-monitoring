import csv
import os
from datetime import datetime

LOG_PATH = "data/monitoring_logs.csv"

FIELDNAMES = [
    "timestamp",
    "model_version",
    "sales",
    "quantity",
    "prediction",
    "latency_seconds",
    "feedback_score",
    "feedback_comment"
]

def init_log_file():
    """Create the log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_PATH):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def log_prediction(
    model_version: str,
    sales: float,
    quantity: int,
    prediction: float,
    latency_seconds: float,
    feedback_score: int,
    feedback_comment: str
):
    """Append one prediction record to monitoring_logs.csv."""
    init_log_file()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_version": model_version,
        "sales": sales,
        "quantity": quantity,
        "prediction": prediction,
        "latency_seconds": round(latency_seconds, 6),
        "feedback_score": feedback_score,
        "feedback_comment": feedback_comment.strip()
    }

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
