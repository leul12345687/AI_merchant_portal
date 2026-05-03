import numpy as np
from PIL import Image
import io
import os
import threading

try:
    import tensorflow as tf
except Exception:
    tf = None

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
    if tf is None:
        print("TensorFlow not available. Image validation AI model is disabled.")
        return None

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
    if tf is None:
        raise RuntimeError("TensorFlow is not installed in this environment")

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
    if tf is None or model is None:
        raise RuntimeError("Image validation is temporarily unavailable on this deployment")

    image = prepare_image(file_bytes)

    with model_lock:
        preds = model.predict(image)

    # Case 1: Custom binary-trained model (single output neuron)
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


def _binary_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = float((y_pred == y_true).mean()) if y_true.size else 0.0
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_true, y_prob)
    auc = float(auc_metric.result().numpy())

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    }


def evaluate_asset_cnn(dataset_dir, batch_size=32, limit=None):
    if tf is None or model is None:
        raise RuntimeError("TensorFlow or model is not available for evaluation")

    if not os.path.isdir(dataset_dir):
        raise ValueError("Dataset directory not found")

    if model.output_shape[-1] == 1:
        label_mode = "binary"
    else:
        label_mode = "categorical"

    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False
    )

    ds = ds.map(
        lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(tf.cast(x, tf.float32)), y)
    )

    num_samples = 0

    if model.output_shape[-1] == 1:
        y_true = []
        y_prob = []

        for images, labels in ds:
            preds = model.predict_on_batch(images).reshape(-1)
            y_prob.extend(preds.tolist())
            y_true.extend(labels.numpy().reshape(-1).tolist())
            num_samples += len(labels)
            if limit and num_samples >= limit:
                break

        if limit:
            y_true = y_true[:limit]
            y_prob = y_prob[:limit]

        metrics = _binary_classification_metrics(y_true, y_prob)
        return {
            "status": "success",
            "num_samples": int(len(y_true)),
            "metrics": metrics
        }

    num_classes = len(ds.class_names)
    if model.output_shape[-1] != num_classes:
        raise RuntimeError("Dataset class count does not match model output classes")

    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_metric = tf.keras.metrics.Mean()

    for images, labels in ds:
        preds = model.predict_on_batch(images)
        top1.update_state(labels, preds)
        top5.update_state(labels, preds)
        loss_metric.update_state(loss_fn(labels, preds))
        num_samples += len(labels)
        if limit and num_samples >= limit:
            break

    return {
        "status": "success",
        "num_samples": int(num_samples if not limit else min(num_samples, limit)),
        "metrics": {
            "top1_accuracy": float(top1.result().numpy()),
            "top5_accuracy": float(top5.result().numpy()),
            "categorical_crossentropy": float(loss_metric.result().numpy())
        }
    }