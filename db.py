import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
# ===============================
# LOAD ENV VARIABLES
# ===============================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

if not MONGO_URI or not DB_NAME:
    raise ValueError("Missing MongoDB environment variables in .env file")

# ===============================
# CREATE MONGO CONNECTION
# ===============================
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Test connection
    client.admin.command("ping")
    print("MongoDB Atlas connected successfully.")

except Exception as e:
    raise ConnectionError(f"MongoDB connection failed: {e}")

# ===============================
# COLLECTIONS
# ===============================
bookings_collection = db["bookings"]


# ===============================
# DATA ACCESS FUNCTIONS
# ===============================

def get_all_bookings():
    """
    Fetch all confirmed + paid bookings
    Used for AI training.
    """
    return list(bookings_collection.find({
        "status": "CONFIRMED",
        "paymentStatus": "PAID"
    }))
def get_bookings_by_merchant(merchant_id):
    try:
        merchant_object_id = ObjectId(merchant_id)
    except Exception:
        return []

    return list(bookings_collection.find({
        "merchant": merchant_object_id,
        "paymentStatus": "PAID",
        "status": "CONFIRMED"
    }))


def get_recent_bookings(days=30):
    """
    Fetch recent bookings for retraining.
    """
    from datetime import datetime, timedelta

    date_limit = datetime.utcnow() - timedelta(days=days)

    return list(bookings_collection.find({
        "createdAt": {"$gte": date_limit},
        "status": "CONFIRMED",
        "paymentStatus": "PAID"
    }))
