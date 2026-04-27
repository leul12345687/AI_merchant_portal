# Merchant AI Demand Advisor Service

This project is a FastAPI-based AI backend for a rental marketplace. It provides:

- Pre-upload demand prediction by asset category.
- Merchant financial analytics and next-month income prediction.
- AI-based uploaded image validation for rental assets.
- Scheduled auto-retraining and keep-alive jobs.

The service is designed for production-style deployment with MongoDB, saved model artifacts, and periodic model refresh.

## 1. System Overview

### Core business goal
Help merchants make better decisions before listing assets by answering:

- Is this uploaded image likely to represent a rentable asset?
- Is demand for this category high, moderate, or low right now?
- How is my income performing, and what might next month look like?

### Main modules

- `main.py`: FastAPI entry point, routes, scheduler jobs, health endpoint.
- `predictdemand.py`: demand model loading + feature computation + category demand inference.
- `train_model.py`: demand model retraining pipeline.
- `financial_ai_service.py`: merchant income analytics + hybrid prediction logic.
- `train_income_model.py`: income model retraining pipeline.
- `asset_cnn_model.py`: image validation model (custom or ImageNet fallback).
- `ai_model_service.py`: standardized wrapper for image validation responses.
- `db.py`: MongoDB connection and booking data access helpers.

## 2. How the Models Work

## 2.1 Demand Prediction Model (Category-level)

### Purpose
Predict likely demand strength for a category before upload.

### Data pipeline

1. Read `assets` collection to map assets to categories.
2. Read `bookings` collection and join bookings to category via asset ID.
3. Build engineered features per category for recent windows.

### Features used

- `total_last_30`: number of bookings in last 30 days.
- `total_units`: sum of booked units in last 30 days.
- `avg_price`: average booking price in last 30 days.
- `growth`: bookings(last 30 days) - bookings(previous 30 days).

### Training algorithm

- Model: `RandomForestRegressor` (`n_estimators=300`, `max_depth=10`).
- Preprocessing: `MinMaxScaler`.
- Evaluation: Mean Absolute Error (MAE) from train/test split.

### Why this algorithm

- Strong for tabular, non-linear relationships.
- Works reliably with limited feature count and mixed scales.
- Less sensitive to noise/outliers than many linear methods.
- Good baseline for retrained, periodic business data.

### Inference output mapping
Numeric prediction is converted to business-friendly levels:

- `>= 50`: High demand.
- `>= 20 and < 50`: Moderate demand.
- `< 20`: Low demand.

Each response includes merchant guidance (`recommended_action`) for operational decisions.

## 2.2 Merchant Income Prediction Model

### Purpose
Estimate merchant-level financial performance and predict next-month income.

### Data pipeline

1. Read paid+confirmed bookings.
2. Aggregate by `(merchant, year, month)`.
3. Build numeric features and train a neural regression model.

### Features used in training

- `netAmount`
- `vatAmount`
- `pricePerUnit`
- `totalPrice`
- `numberOfUnits`

Target:

- Monthly `netAmount` (income).

### Training algorithm

- Model: TensorFlow/Keras feed-forward neural network.
  - Dense(64, ReLU)
  - Dense(32, ReLU)
  - Dense(1)
- Loss: Mean Squared Error (MSE)
- Metric: MAE
- Optimizer: Adam
- Preprocessing: `MinMaxScaler`

### Why this algorithm

- Can model non-linear income behavior from interacting numeric features.
- Lightweight architecture keeps training/inference manageable.
- Easy to retrain periodically with new transactional data.

### Hybrid prediction logic in production
`financial_ai_service.py` combines:

1. Statistical signal:
   - historical average
   - last 3-month average
   - linear trend slope (`np.polyfit`)
   - current month projection
2. AI model estimate (if model/scaler available).
3. Weighted fusion:

$$
\text{predicted\_next\_month} = 0.6 \cdot \text{statistical\_prediction} + 0.4 \cdot \text{ai\_prediction}
$$

Then safety-floor logic prevents unrealistically low forecasts:

$$
\text{minimum\_allowed} = 0.6 \cdot \text{historical\_average}
$$

Final prediction is bounded by this floor.

## 2.3 Asset Image Validation Model

### Purpose
Prevent non-rental images from being uploaded.

### Runtime strategy

- If `rental_validator.keras` exists: load and run custom model.
- Else: fallback to `EfficientNetB0` pretrained on ImageNet.

### Inference logic

- Custom binary model: sigmoid output threshold at `0.5`.
- Fallback model: decode top ImageNet class and match against rental-related keywords (`truck`, `tractor`, `camera`, `tool`, etc.).

### Why this approach

- Robust startup even without custom artifact.
- Transfer learning fallback keeps service functional.
- Thread-safe prediction via lock avoids concurrent inference race conditions.

## 3. Libraries and Technology Stack

From `requirements.txt`:

- `fastapi`, `uvicorn`, `python-multipart`: REST API + file upload.
- `python-dotenv`: environment variable management.
- `tensorflow-cpu`: deep learning models (income + image validation fallback).
- `scikit-learn`: random forest, scaling, model evaluation.
- `pandas`, `numpy`: preprocessing, feature engineering, analytics math.
- `joblib`, `pickle`: persistence of scaler/model artifacts.
- `pymongo`: MongoDB integration.
- `apscheduler`: periodic retraining and keep-alive jobs.
- `pillow`: image decoding and preprocessing.
- `requests`: external keep-alive pings.

Runtime target:

- Python `3.10.13` (`runtime.txt`).

## 4. Datasets Used

This system uses operational MongoDB data (not a static CSV dataset).

### Primary collections

- `bookings`
- `assets`

### Effective training/inference data rules

- Income training relies on bookings filtered as:
  - `status = CONFIRMED`
  - `paymentStatus = PAID`
- Demand features use booking history by category over last 60 days.
- Income model uses merchant-month aggregates from approved payment timestamps.

### Image model dataset source

- Custom dataset details are external to this repository if `rental_validator.keras` was trained elsewhere.
- If fallback mode is active, knowledge comes from ImageNet-pretrained EfficientNetB0.

For thesis reporting, include a data governance subsection with:

- Data source (production transactional DB).
- Filters and cleaning logic.
- Handling of missing dates/values.
- Privacy and access controls.

## 5. API Endpoints

- `GET /health`
  - Service health check.

- `GET /merchant-income/{merchant_id}`
  - Returns income analytics, tax estimate, and predicted next month.

- `POST /validate-asset-image`
  - Multipart image upload for rental relevance validation.

- `GET /pre-upload-demand?category=...`
  - Returns predicted demand level and merchant action recommendation.

## 6. Auto-Retraining and MLOps Behavior

Scheduled in `main.py` via APScheduler:

- Demand model retraining every 6 hours (`train_predict_model`).
- Income model retraining every 6 hours (`train_model`).
- Keep-alive pings every 10 minutes to configured external services.

Demand model artifacts are saved atomically (`.tmp` + `os.replace`) to avoid partial-write corruption.

`predictdemand.py` dynamically reloads model/scaler if artifact modification time changes, enabling near-zero-downtime model refresh.

## 7. How to Run

## 7.1 Install dependencies

```bash
pip install -r requirements.txt
```

## 7.2 Configure environment

Create `.env` with:

```env
MONGO_URI=your_mongodb_uri
DB_NAME=your_database_name
MODEL_PATH=demand_model.pkl
SCALER_PATH=scaler.pkl
KEEP_ALIVE_URLS=https://service-a.example.com,https://service-b.example.com
```

Security note: do not commit real credentials to version control.

## 7.3 Start API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 7.4 Deploy on Render (important)

To avoid deployment failures:

- Use Python `3.10.13` (defined in `.python-version` and `render.yaml`).
- Use build command:

```bash
pip install --upgrade pip; pip install -r requirements.txt
```

- Do **not** use `pip && pip install -r requirements.txt`.
- Use start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Note: `tensorflow-cpu` is constrained to Python `< 3.12` in `requirements.txt`.
On environments without TensorFlow, the API still starts and falls back gracefully for income prediction and image validation.

## 8. Model Artifacts in Repository

- `demand_model.pkl`
- `demand_model.pkl.meta`
- `scaler.pkl`
- `income_prediction_model.keras`
- `income_scaler.save`

These artifacts represent previously trained model states and can be replaced by scheduled retraining.

## 9. Recommended Thesis Content (Must Include)

Use the following structure in your final project thesis.

1. Problem definition and business motivation.
2. System architecture (API, DB, model services, scheduler).
3. Data sources and preprocessing pipeline.
4. Feature engineering design for each model.
5. Algorithm selection rationale:
   - Random Forest for demand.
   - Dense neural regression for income.
   - EfficientNet/custom binary approach for image validation.
6. Training strategy and evaluation metrics (MAE, loss/MAE curves).
7. Inference-time business logic and decision thresholds.
8. MLOps and deployment:
   - model persistence,
   - atomic save,
   - scheduled retraining,
   - model hot-reload.
9. API contract and integration with frontend/merchant workflows.
10. Limitations, risks, ethics, and future improvements.

## 10. Current Limitations and Future Improvements

### Limitations

- Demand target currently mirrors recent booking volume, so long-horizon forecasting may be limited.
- Small feature set may underfit complex market dynamics.
- Image fallback uses keyword-matched ImageNet labels, which is approximate.

### Suggested improvements

- Add richer demand features (seasonality, location, holidays, cancellations).
- Track offline metrics over time and alert on model drift.
- Introduce model registry/versioning and canary rollout.
- Build a fully supervised rental-image dataset with domain-specific labels.

---

If used in academic submission, include screenshots of API tests, model training logs, and architecture diagrams to strengthen reproducibility and evaluation clarity.