import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os
import threading

# ==============================
# Configuration
# ==============================
MODEL_PATH = "rental_validator.pt"
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
        print("Loading trained rental validator model (PyTorch)...")
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Binary output
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model

    print("No trained model found. Using pretrained ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_model_safe()

# ==============================
# Image Transform Pipeline
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# Prepare uploaded image
# ==============================
def prepare_image(file_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = transform(img).unsqueeze(0)
        return img
    except Exception:
        raise ValueError("Invalid image file")

# ==============================
# Validate asset image
# ==============================
def validate_asset_image(file_bytes: bytes):
    image = prepare_image(file_bytes)

    with model_lock:
        with torch.no_grad():
            preds = model(image)

    # Binary classification (ResNet output)
    score = torch.sigmoid(preds).item()
    is_rental = score > 0.5

    return {
        "allowed_upload": bool(is_rental),
        "confidence": float(score if is_rental else 1 - score),
        "message": "Rental asset image accepted"
        if is_rental else "Rejected: non-rental image"
    }