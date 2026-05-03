# ===============================================
# Income Prediction Training Model
# ===============================================
# - Reads booking data from MongoDB
# - Aggregates monthly income per merchant
# - Learns merchant income patterns
# - Saves trained model + scaler
# - Used for auto-retraining
# ===============================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
from db import get_all_bookings

try:
    import tensorflow as tf
except Exception:
    tf = None


MODEL_PATH = "income_prediction_model.keras"
SCALER_PATH = "income_scaler.save"


# -----------------------------------
# Safe date parser
# -----------------------------------
def parse_payment_date(approved):
    if not approved:
        return None

    if isinstance(approved, datetime):
        return approved

    if isinstance(approved, str):
        try:
            return datetime.fromisoformat(approved.replace("Z", "+00:00"))
        except Exception:
            return None

    return None


# -----------------------------------
# Prepare training dataset
# -----------------------------------
def _safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _safe_mape(y_true, y_pred)
    }


def _build_income_dataset():
    bookings = get_all_bookings()

    if len(bookings) == 0:
        raise ValueError("No bookings found for training")

    data = []

    for booking in bookings:
        payment_date = parse_payment_date(booking.get("paymentApprovedAt"))
        if not payment_date:
            continue  # skip unapproved or invalid payments

        data.append({
            "merchant": str(booking["merchant"]),
            "netAmount": booking.get("netAmount", 0),
            "vatAmount": booking.get("vatAmount", 0),
            "pricePerUnit": booking.get("pricePerUnit", 0),
            "totalPrice": booking.get("totalPrice", 0),
            "numberOfUnits": booking.get("numberOfUnits", 1),
            "year": payment_date.year,
            "month": payment_date.month
        })

    df = pd.DataFrame(data)

    # Aggregate monthly per merchant
    df_agg = df.groupby(["merchant", "year", "month"]).agg({
        "netAmount": "sum",
        "vatAmount": "sum",
        "pricePerUnit": "mean",
        "totalPrice": "sum",
        "numberOfUnits": "sum"
    }).reset_index()

    # Features and target
    features = df_agg[[
        "netAmount",
        "vatAmount",
        "pricePerUnit",
        "totalPrice",
        "numberOfUnits"
    ]]

    target = df_agg["netAmount"]  # monthly income target

    return features, target


def prepare_training_data():
    features, target = _build_income_dataset()

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    joblib.dump(scaler, SCALER_PATH)

    return X_scaled, target


# -----------------------------------
# Train regression model
# -----------------------------------
def train_model():
    if tf is None:
        print("TensorFlow is not installed. Skipping income model training.")
        return

    X, y = prepare_training_data()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    model.fit(
        X,
        y,
        epochs=20,
        batch_size=16,
        verbose=1
    )

    model.save(MODEL_PATH, save_format="keras")

    print("Income model trained & saved successfully")


def evaluate_income_model(test_size=0.2, random_state=42):
    if tf is None:
        return {"status": "error", "message": "TensorFlow is not installed."}

    try:
        features, target = _build_income_dataset()
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if features.empty:
        return {"status": "error", "message": "No valid booking data after preprocessing."}

    if not os.path.exists(MODEL_PATH):
        return {"status": "error", "message": "Income model not found."}

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(features)
    else:
        X_scaled = MinMaxScaler().fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        target,
        test_size=test_size,
        random_state=random_state
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict(X_test, verbose=0).reshape(-1)
    metrics = _regression_metrics(y_test, preds)

    return {
        "status": "success",
        "num_samples": int(len(features)),
        "test_size": float(test_size),
        "metrics": metrics
    }


# -----------------------------------
# Run training if executed directly
# -----------------------------------
if __name__ == "__main__":
    train_model()