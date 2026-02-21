from fastapi import FastAPI, UploadFile, File, HTTPException
from ai_model_service import check_uploaded_asset

import asyncio
from fastapi import FastAPI, HTTPException
from predictdemand import predict_pre_upload_demand

from financial_ai_service import calculate_financials
from train_income_model import train_model
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from train_model import train_predict_model
import pickle
import time
# ==============================
# Load environment variables
# ==============================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")

# ==============================
# MongoDB Connection
# ==============================
if not MONGO_URI or not DB_NAME:
    raise ValueError("MongoDB environment variables not set!")

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


# -----------------------------------------
# Health check endpoint
# -----------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Rental Asset Fraud Detection API"
    }

scheduler = BackgroundScheduler()

# Run demand model every 6 hours
scheduler.add_job(train_predict_model, 'interval', hours=6, next_run_time=None)

# Run income model every 6 hours
scheduler.add_job(train_model, 'interval', hours=6, next_run_time=None)

scheduler.start()



@app.get("/merchant-income/{merchant_id}")
def merchant_income(merchant_id: str):

    result = calculate_financials(merchant_id)

    return {
        "status": "success",
        "data": result
    }
# -----------------------------------------
# Image validation endpoint
# -----------------------------------------
@app.post("/validate-asset-image")
async def validate_asset_image(file: UploadFile = File(...)):
    """
    Merchant uploads asset image.
    AI checks if image is valid rental asset or not.
    """

    try:
        # read uploaded image
        contents = await file.read()

        # send to AI service
        result = check_uploaded_asset(contents)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image validation failed: {str(e)}"
        )
    
@app.on_event("startup")
def startup_event():
    # No need to reload model here; asset_cnn_model loads it on import
    print("API starting up...")


# =========================================
# PRE-UPLOAD DEMAND ADVISOR FOR MERCHANT
# =========================================
@app.get("/pre-upload-demand")
def pre_upload_demand(category: str):
    """
    Called BEFORE merchant uploads an asset.
    Returns demand level and actionable advice based on latest trained model.
    """
    try:
        # prediction function already loads latest model atomically
        result = predict_pre_upload_demand(category)
        result["category"] = category
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")