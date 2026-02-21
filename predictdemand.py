# ===============================================
# SMART DEMAND PREDICTION SERVICE
# Category-based | Pre-upload merchant intelligence
# ===============================================

import os
import pickle
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
import time
import threading
_model_lock = threading.Lock()
# ==============================
# Logging configuration
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================
# Load environment variables
# ==============================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")

if not MONGO_URI or not DB_NAME or not MODEL_PATH or not SCALER_PATH:
    raise ValueError("MongoDB or model/scaler paths not set!")

# ==============================
# MongoDB Connection
# ==============================
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ==============================
# Dynamic model loader
# ==============================
_model = None
_scaler = None
_model_mtime = None

def load_model_if_updated():
    global _model, _scaler, _model_mtime
    with _model_lock:
        try:
            mtime = os.path.getmtime(MODEL_PATH)
            if _model is None or _model_mtime != mtime:
                with open(MODEL_PATH, "rb") as f:
                    _model = pickle.load(f)
                with open(SCALER_PATH, "rb") as f:
                    _scaler = pickle.load(f)
                _model_mtime = mtime
                logging.info("✅ Model and scaler loaded/reloaded.")
        except Exception as e:
            logging.error(f"Failed to load model/scaler: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
# ==============================
# Feature Engineering
# ==============================
def compute_category_features(category: str) -> list:
    """
    Compute features for a category:
    - total bookings last 30 days
    - total units
    - average price
    - growth vs previous 30 days
    """
    now = pd.Timestamp.now()
    cutoff_30 = now - pd.Timedelta(days=30)
    cutoff_60 = now - pd.Timedelta(days=60)

    # Assets in this category
    assets = list(db.assets.find({"category": category}, {"_id": 1}))
    if not assets:
        return None

    asset_ids = [a["_id"] for a in assets]

    bookings = list(db.bookings.find({
        "asset": {"$in": asset_ids},
        "paymentApprovedAt": {"$gte": cutoff_60.to_pydatetime()}
    }))

    if not bookings:
        return [0, 0, 0, 0]

    df = pd.DataFrame(bookings)
    for col in ["paymentApprovedAt", "numberOfUnits", "totalPrice"]:
        if col not in df.columns:
            df[col] = 0

    df["paymentApprovedAt"] = pd.to_datetime(df["paymentApprovedAt"], errors="coerce")
    df = df.dropna()

    last_30 = df[df["paymentApprovedAt"] >= cutoff_30]
    prev_30 = df[(df["paymentApprovedAt"] < cutoff_30) & (df["paymentApprovedAt"] >= cutoff_60)]

    total_last_30 = len(last_30)
    total_units = last_30["numberOfUnits"].sum() if not last_30.empty else 0
    avg_price = last_30["totalPrice"].mean() if not last_30.empty else 0
    growth = total_last_30 - len(prev_30)

    return [total_last_30, total_units, avg_price, growth]

# ==============================
# Core AI - Pre-upload demand
# ==============================
def predict_pre_upload_demand(category: str) -> dict:
    """
    Returns predicted demand for a given category with actionable advice.
    Auto-reloads latest model if updated.
    """
    load_model_if_updated()  # ensures latest model

    features = compute_category_features(category)

    if features is None:
        logging.info(f"No data for category: {category}")
        return {
            "category": category,
            "demand_level": "Unknown",
            "merchant_notification": "No historical data available.",
            "recommended_action": "Upload cautiously and gather data."
        }

    try:
        X_scaled = _scaler.transform([features])
        pred_value = float(_model.predict(X_scaled)[0])
    except Exception as e:
        logging.error(f"Prediction failed for {category}: {e}")
        pred_value = 0

    # Map numeric prediction → demand level
    if pred_value >= 50:
        demand_level = "High"
        notification = "High bookings detected last 30 days."
        action = "Upload immediately to capture demand."
    elif pred_value >= 20:
        demand_level = "Moderate"
        notification = "Moderate demand observed."
        action = "Upload with competitive pricing."
    else:
        demand_level = "Low"
        notification = "Low demand detected."
        action = "Consider marketing, discounts, or alternative categories."

    result = {
        "category": category,
        "predicted_demand_value": pred_value,
        "demand_level": demand_level,
        "merchant_notification": notification,
        "recommended_action": action,
        "feature_snapshot": {
            "last_30_day_bookings": features[0],
            "growth_trend": features[3]
        }
    }

    logging.info(f"Prediction for {category}: {result}")
    return result
