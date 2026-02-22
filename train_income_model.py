"""
Income Prediction Training Model (PyTorch Version)
---------------------------------------------------
- Reads booking data from MongoDB
- Aggregates monthly income per merchant
- Learns income patterns using PyTorch
- Saves model + scaler
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from db import get_all_bookings

MODEL_PATH = "income_prediction_model.pt"
SCALER_PATH = "income_scaler.save"

# -------------------------------
# Safe date parser
# -------------------------------
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

# -------------------------------
# Prepare dataset
# -------------------------------
def prepare_training_data():
    bookings = get_all_bookings()
    if len(bookings) == 0:
        raise ValueError("No bookings found for training")

    data = []
    for b in bookings:
        payment_date = parse_payment_date(b.get("paymentApprovedAt"))
        if not payment_date:
            continue  # skip unapproved payments

        data.append({
            "merchant": str(b["merchant"]),
            "netAmount": b.get("netAmount", 0),
            "vatAmount": b.get("vatAmount", 0),
            "pricePerUnit": b.get("pricePerUnit", 0),
            "totalPrice": b.get("totalPrice", 0),
            "numberOfUnits": b.get("numberOfUnits", 1),
            "year": payment_date.year,
            "month": payment_date.month
        })

    df = pd.DataFrame(data)

    # Aggregate by merchant + year + month
    df_agg = df.groupby(["merchant", "year", "month"]).agg({
        "netAmount": "sum",
        "vatAmount": "sum",
        "pricePerUnit": "mean",
        "totalPrice": "sum",
        "numberOfUnits": "sum"
    }).reset_index()

    # Features & target
    features = df_agg[["netAmount", "vatAmount", "pricePerUnit", "totalPrice", "numberOfUnits"]]
    target = df_agg["netAmount"].values.reshape(-1, 1)  # monthly income

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)
    y_scaled = scaler.fit_transform(target)

    joblib.dump(scaler, SCALER_PATH)

    return X_scaled, y_scaled

# -------------------------------
# PyTorch Regression Model
# -------------------------------
class IncomeModel(nn.Module):
    def __init__(self, input_size):
        super(IncomeModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# -------------------------------
# Train Model
# -------------------------------
def train_model():
    X, y = prepare_training_data()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = IncomeModel(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Income model trained & saved successfully")

# -------------------------------
# Run training
# -------------------------------
if __name__ == "__main__":
    train_model()