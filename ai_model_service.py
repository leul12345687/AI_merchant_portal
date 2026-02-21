from asset_cnn_model import validate_asset_image

# -----------------------------------------
# AI Service Layer
# Used by FastAPI endpoints or other modules
# -----------------------------------------

def check_uploaded_asset(file_bytes: bytes):
    """
    Validate a merchant-uploaded image using the AI rental validator model.
    Prevent non-rental image uploads.
    """

    try:
        # Use the main model function
        result = validate_asset_image(file_bytes)

        # Standardized response
        response = {
            "status": "success",
            "allowed_upload": result.get("allowed_upload", False),
            "confidence": result.get("confidence", 0.0),
            "prediction": result.get("prediction", "unknown"),
            "message": result.get("message", "")
        }

        return response

    except Exception as e:
        # Catch errors and return a consistent error structure
        return {
            "status": "error",
            "allowed_upload": False,
            "confidence": 0.0,
            "prediction": "unknown",
            "message": f"AI validation failed: {str(e)}"
        }
