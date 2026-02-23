# ===============================================
# Financial AI Service - Advanced Version
# ===============================================
# Features:
# - Monthly income (current month)
# - Yearly income (all historical)
# - Smart next month prediction (history + trend + AI)
# - Tax estimation
# - Profit calculation
# ===============================================

import os
import tensorflow as tf
import numpy as np
import joblib
from datetime import datetime
import calendar
from collections import defaultdict
from db import get_bookings_by_merchant


# ===============================================
# Configuration
# ===============================================
MODEL_PATH = "income_prediction_model.keras"
SCALER_PATH = "income_scaler.save"
TAX_RATE = 0.15


# ===============================================
# Safe Model Loader
# ===============================================
model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Income prediction model loaded successfully.")
    except Exception as e:
        print("Error loading income model:", e)
else:
    print("Income model or scaler not found. Prediction will use statistical logic.")


# ===============================================
# Safe Date Parser
# ===============================================
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


# ===============================================
# Group Income by Month
# ===============================================
def group_income_by_month(bookings):
    monthly_income = defaultdict(float)

    for booking in bookings:
        payment_date = parse_payment_date(booking.get("paymentApprovedAt"))
        if not payment_date:
            continue

        key = (payment_date.year, payment_date.month)
        monthly_income[key] += booking.get("netAmount", 0)

    sorted_months = sorted(monthly_income.items())
    return [income for _, income in sorted_months]


# ===============================================
# Trend Calculation (Linear Regression)
# ===============================================
def calculate_trend(monthly_incomes):
    if len(monthly_incomes) < 2:
        return 0

    x = np.arange(len(monthly_incomes))
    y = np.array(monthly_incomes)
    slope = np.polyfit(x, y, 1)[0]
    return slope


# ===============================================
# Main Financial Calculation
# ===============================================
def calculate_financials(merchant_id: str):

    bookings = get_bookings_by_merchant(merchant_id)

    if not bookings:
        return {
            "status": "error",
            "message": "No booking data available"
        }

    now = datetime.utcnow()

    # -------------------------------------------
    # Current Month Income
    # -------------------------------------------
    current_month_income = 0

    for booking in bookings:
        payment_date = parse_payment_date(booking.get("paymentApprovedAt"))
        if (
            payment_date
            and payment_date.year == now.year
            and payment_date.month == now.month
        ):
            current_month_income += booking.get("netAmount", 0)

    # -------------------------------------------
    # Yearly Income (All Historical)
    # -------------------------------------------
    yearly_income = sum(
        booking.get("netAmount", 0)
        for booking in bookings
    )

    # -------------------------------------------
    # Historical Monthly Learning
    # -------------------------------------------
    monthly_incomes = group_income_by_month(bookings)

    historical_avg = (
        float(np.mean(monthly_incomes))
        if monthly_incomes else 0
    )

    last_3_avg = (
        float(np.mean(monthly_incomes[-3:]))
        if len(monthly_incomes) >= 3
        else historical_avg
    )

    trend_slope = calculate_trend(monthly_incomes)

    # -------------------------------------------
    # Current Month Projection
    # -------------------------------------------
    days_in_month = calendar.monthrange(now.year, now.month)[1]

    if now.day > 0:
        projected_current_month = (
            (current_month_income / now.day) * days_in_month
        )
    else:
        projected_current_month = current_month_income

    # -------------------------------------------
    # Statistical Base Prediction
    # -------------------------------------------
    if monthly_incomes:
        trend_projection = monthly_incomes[-1] + trend_slope
    else:
        trend_projection = projected_current_month

    statistical_prediction = (
        0.4 * trend_projection +
        0.3 * last_3_avg +
        0.3 * projected_current_month
    )

    # -------------------------------------------
    # AI Model Enhancement (Optional)
    # -------------------------------------------
    ai_prediction = 0

    if model is not None and scaler is not None and len(bookings) > 0:
        vat_total = sum(b.get("vatAmount", 0) for b in bookings)
        avg_price_per_unit = np.mean([b.get("pricePerUnit", 0) for b in bookings])
        avg_total_price = np.mean([b.get("totalPrice", 0) for b in bookings])
        avg_units = np.mean([b.get("numberOfUnits", 1) for b in bookings])

        sample = np.array([[
            projected_current_month,
            vat_total,
            avg_price_per_unit,
            avg_total_price,
            avg_units
        ]])

        try:
            sample_scaled = scaler.transform(sample)
            ai_prediction = model.predict(sample_scaled, verbose=0)[0][0]
        except Exception:
            ai_prediction = statistical_prediction

    # -------------------------------------------
    # Final Hybrid Prediction
    # -------------------------------------------
    if ai_prediction > 0:
        predicted_next_month = (
            0.6 * statistical_prediction +
            0.4 * ai_prediction
        )
    else:
        predicted_next_month = statistical_prediction

    # Safety Floor Protection
    minimum_allowed = historical_avg * 0.6

    if predicted_next_month < minimum_allowed:
        predicted_next_month = minimum_allowed

    # -------------------------------------------
    # Tax & Profit Calculation
    # -------------------------------------------
    estimated_tax = yearly_income * TAX_RATE
    profit_after_tax = yearly_income - estimated_tax

    # -------------------------------------------
    # Final Response
    # -------------------------------------------
    return {
        "status": "success",
        "monthly_income": round(current_month_income, 2),
        "projected_current_month": round(projected_current_month, 2),
        "historical_monthly_average": round(historical_avg, 2),
        "trend_slope": round(trend_slope, 2),
        "yearly_income": round(yearly_income, 2),
        "predicted_next_month": round(float(predicted_next_month), 2),
        "estimated_tax_year": round(estimated_tax, 2),
        "profit_after_tax": round(profit_after_tax, 2),
        "months_used_for_learning": len(monthly_incomes)
    }