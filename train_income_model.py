# ===============================================
# Income Prediction Training Model
# ===============================================
# - Reads booking data from MongoDB
# - Aggregates monthly income per merchant
# - Learns merchant income patterns
# - Saves trained model + scaler
# - Used for auto-retraining
# ===============================================

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
from db import get_all_bookings


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
def prepare_training_data():
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

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    joblib.dump(scaler, SCALER_PATH)

    return X_scaled, target


# -----------------------------------
# Train regression model
# -----------------------------------
def train_model():
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

    model.save(MODEL_PATH)

    print("Income model trained & saved successfully")


# -----------------------------------
# Run training if executed directly
# -----------------------------------
if __name__ == "__main__":
    train_model()