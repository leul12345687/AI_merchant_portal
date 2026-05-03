# ===============================================
# AUTO-RETRAIN DEMAND MODEL SERVICE - ATOMIC SAVE ONLY
# ===============================================
import os
import logging
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from dotenv import load_dotenv

# ==============================
# Logging
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==============================
# Load environment variables
# ==============================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ==============================
# Atomic save utility
# ==============================
def atomic_save(obj, file_path):
    temp_path = file_path + ".tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(temp_path, file_path)
    logging.info(f"✅ Saved {file_path} atomically.")

# ==============================
# Core training pipeline
# ==============================
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


def _build_demand_dataset():
    bookings = list(db.bookings.find())
    assets = list(db.assets.find())
    if not bookings or not assets:
        return None, None, "Not enough data to train/evaluate."

    asset_dict = {str(a["_id"]): a.get("category", "Unknown") for a in assets}

    data_rows = []
    for b in bookings:
        asset_id = str(b.get("asset"))
        category = asset_dict.get(asset_id, "Unknown")
        payment_date = pd.to_datetime(b.get("paymentApprovedAt"), errors="coerce")
        if pd.isna(payment_date):
            continue
        data_rows.append({
            "asset_id": asset_id,
            "category": category,
            "paymentApprovedAt": payment_date,
            "totalPrice": b.get("totalPrice", 0),
            "numberOfUnits": b.get("numberOfUnits", 0)
        })

    df = pd.DataFrame(data_rows)
    if df.empty:
        return None, None, "No valid booking data after preprocessing."

    def compute_features(group_df):
        now = pd.Timestamp.now()
        last_30 = group_df[group_df["paymentApprovedAt"] >= (now - pd.Timedelta(days=30))]
        prev_30 = group_df[(group_df["paymentApprovedAt"] < (now - pd.Timedelta(days=30))) &
                           (group_df["paymentApprovedAt"] >= (now - pd.Timedelta(days=60)))]
        total_last_30 = len(last_30)
        total_units = last_30["numberOfUnits"].sum() if not last_30.empty else 0
        avg_price = last_30["totalPrice"].mean() if not last_30.empty else 0
        growth = total_last_30 - len(prev_30)
        return [total_last_30, total_units, avg_price, growth]

    X, y = [], []
    for _, group in df.groupby("category"):
        feats = compute_features(group)
        X.append(feats)
        y.append(feats[0])

    X = np.array(X)
    y = np.array(y)
    if len(X) < 1:
        return None, None, "Not enough training samples."

    return X, y, None


def train_predict_model():
    try:
        logging.info("🚀 Starting model training...")
        X, y, error_msg = _build_demand_dataset()
        if error_msg:
            logging.warning(error_msg)
            return False

        # Scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        atomic_save(scaler, SCALER_PATH)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = _regression_metrics(y_test, preds)
        logging.info(
            "Model trained successfully | MAE: %.4f | RMSE: %.4f | R2: %.4f | MAPE: %.2f%%",
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["mape"]
        )

        atomic_save(model, MODEL_PATH)
        meta = {
            "last_trained": pd.Timestamp.now().isoformat(),
            "num_samples": len(X),
            "metrics": metrics
        }
        atomic_save(meta, MODEL_PATH + ".meta")

        return True

    except Exception as e:
        logging.error(f"Training failed: {e}")
        return False


def evaluate_predict_model(test_size=0.2, random_state=42):
    try:
        X, y, error_msg = _build_demand_dataset()
        if error_msg:
            return {"status": "error", "message": error_msg}

        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return {"status": "error", "message": "Model or scaler not found."}

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        try:
            X_scaled = scaler.transform(X)
        except Exception:
            X_scaled = MinMaxScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=random_state
        )

        preds = model.predict(X_test)
        metrics = _regression_metrics(y_test, preds)

        return {
            "status": "success",
            "num_samples": int(len(X)),
            "test_size": float(test_size),
            "metrics": metrics
        }
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    success = train_predict_model()
    if success:
        print("✅ Model training completed successfully.")
    else:
        print("⚠️ Model training failed or not enough data.")    