from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from ai_model_service import check_uploaded_asset
from predictdemand import predict_pre_upload_demand
from financial_ai_service import calculate_financials
from train_income_model import train_model, evaluate_income_model
from train_model import train_predict_model, evaluate_predict_model
from asset_cnn_model import evaluate_asset_cnn
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import requests
import threading
import time

# ==============================
# Load environment variables
# ==============================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
KEEP_ALIVE_URLS = os.getenv("KEEP_ALIVE_URLS", "").split(",")

if not MONGO_URI or not DB_NAME:
    raise ValueError("MongoDB environment variables not set!")

# ==============================
# MongoDB Connection
# ==============================
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ==============================
# FastAPI app
# ==============================
app = FastAPI(title="Merchant AI Demand Advisor")

# ==============================
# GLOBAL MODEL CACHE
# ==============================
model = None
scaler = None
model_last_loaded_time = 0

# ==============================
# Background Scheduler
# ==============================
scheduler = BackgroundScheduler()

# ==============================
# Keep-alive / AI Wake-up Thread
# ==============================
def ping_services():
    print("🚀 Running keep-alive job...")

    for url in KEEP_ALIVE_URLS:
        url = url.strip()
        if not url:
            continue

        try:
            full_url = f"{url}/health"
            resp = requests.get(full_url, timeout=10)
            print(f"🔁 Ping {full_url}: {resp.status_code}")
        except Exception as e:
            print(f"⚠️ Failed to ping {url}: {str(e)}")
# Run demand model every 6 hours
scheduler.add_job(train_predict_model, 'interval', hours=6, next_run_time=None)
scheduler.add_job(ping_services, 'interval', minutes=10)
# Run income model every 6 hours
scheduler.add_job(train_model, 'interval', hours=6, next_run_time=None)

scheduler.start()

# ==============================
# Keep-alive / AI Wake-up Thread
# ==============================
def ping_services():
    print("🚀 Running keep-alive job...")

    for url in KEEP_ALIVE_URLS:
        url = url.strip()
        if not url:
            continue

        try:
            full_url = f"{url}/health"
            resp = requests.get(full_url, timeout=10)
            print(f"🔁 Ping {full_url}: {resp.status_code}")
        except Exception as e:
            print(f"⚠️ Failed to ping {url}: {str(e)}")

# ==============================
# Health Check
# ==============================

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}

# ==============================
# Merchant Income Endpoint
# ==============================
@app.get("/merchant-income/{merchant_id}")
def merchant_income(merchant_id: str):
    try:
        result = calculate_financials(merchant_id)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating income: {str(e)}")

# ==============================
# Image Validation Endpoint
# ==============================
@app.post("/validate-asset-image")
async def validate_asset_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = check_uploaded_asset(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image validation failed: {str(e)}")

# ==============================
# Pre-upload Demand Advisor
# ==============================
@app.get("/pre-upload-demand")
def pre_upload_demand(category: str):
    try:
        result = predict_pre_upload_demand(category)
        result["category"] = category
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# ==============================
# Model Evaluation Endpoints
# ==============================

@app.post("/evaluate-demand-model")
def evaluate_demand_model():
    result = evaluate_predict_model()
    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=result.get("message", "Evaluation failed"))
    return result


@app.post("/evaluate-income-model")
def evaluate_income_model_endpoint():
    result = evaluate_income_model()
    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=result.get("message", "Evaluation failed"))
    return result


@app.post("/evaluate-asset-cnn")
def evaluate_asset_cnn_endpoint(payload: dict = Body(...)):
    dataset_dir = payload.get("dataset_dir") or os.getenv("IMAGENET_DATASET_DIR")
    if not dataset_dir:
        raise HTTPException(
            status_code=400,
            detail="dataset_dir is required or set IMAGENET_DATASET_DIR"
        )

    batch_size = payload.get("batch_size", 32)
    limit = payload.get("limit")

    try:
        result = evaluate_asset_cnn(dataset_dir, batch_size=batch_size, limit=limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==============================
# Startup Event
# ==============================
@app.on_event("startup")
def startup_event():
    print("API starting up... Keep-alive thread started, background jobs scheduled.")