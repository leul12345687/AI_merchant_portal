import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import threading

# ==============================
# Configuration
# ==============================
MODEL_PATH = "rental_validator.keras"
IMG_SIZE = 224

# Thread safety for model prediction
model_lock = threading.Lock()

# Keywords to identify rental assets
RENTAL_KEYWORDS = [
    "truck", "tractor", "car", "bus", "bulldozer",
    "house", "building", "apartment", "villa",
    "bed", "chair", "sofa", "table",
    "generator", "speaker", "camera",
    "tool", "machine", "equipment"
]

# ==============================
# Load or initialize model
# ==============================
def load_model_safe():
    if os.path.exists(MODEL_PATH):
        print("Loading trained rental validator model...")
        return tf.keras.models.load_model(MODEL_PATH)

    print("No trained model found. Using pretrained EfficientNetB0...")
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=True
    )
    return base_model

model = load_model_safe()

# ==============================
# Save model if first time using custom training
# ==============================
def save_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Saving model to {MODEL_PATH} ...")
        model.save(MODEL_PATH)
        print("Model saved successfully!")

# ==============================
# Prepare uploaded image
# ==============================
def prepare_image(file_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception:
        raise ValueError("Invalid image file")

# ==============================
# Validate asset image
# ==============================
def validate_asset_image(file_bytes: bytes):
    image = prepare_image(file_bytes)

    with model_lock:
        preds = model.predict(image)

    # Case 1: Custom binary-trained model (output shape 1)
    if model.output_shape[-1] == 1:
        score = preds[0][0]
        is_rental = score > 0.5
        return {
            "allowed_upload": bool(is_rental),
            "confidence": float(score if is_rental else 1 - score),
            "message": "Rental asset image accepted"
            if is_rental else "Rejected: non-rental image"
        }

    # Case 2: Pretrained ImageNet fallback
    decoded = tf.keras.applications.efficientnet.decode_predictions(preds, top=3)[0]
    label = decoded[0][1]
    confidence = float(decoded[0][2])
    is_rental = any(word in label.lower() for word in RENTAL_KEYWORDS)

    return {
        "allowed_upload": bool(is_rental),
        "prediction": label,
        "confidence": confidence,
        "message": "Rental asset image accepted"
        if is_rental else "Rejected: non-rental image"
    }
