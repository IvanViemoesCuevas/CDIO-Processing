import os

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

# Path to your custom YOLO model. Replace this placeholder with your model path.
# Set to None to disable YOLO and always use the contour fallback.
YOLO_MODEL_PATH = "my_model.pt"

# Lazily-loaded YOLO model instance
_yolo_model = None


def _ensure_yolo_model():
    """Lazy-load the YOLO model. Returns None if ultralytics not installed."""
    global _yolo_model
    if YOLO is None:
        return None
    # If user explicitly disabled or didn't set a path, fall back
    if not YOLO_MODEL_PATH:
        print("YOLO model path not set; using contour-based fallback.")
        return None

    # If the file doesn't exist, fall back rather than raising during import
    if not os.path.isfile(YOLO_MODEL_PATH):
        print(f"YOLO model file not found at '{YOLO_MODEL_PATH}'; using contour-based fallback.")
        return None

    if _yolo_model is None:
        try:
            # Instantiate model (user should edit YOLO_MODEL_PATH to point at their weights)
            _yolo_model = YOLO(YOLO_MODEL_PATH)
        except Exception as e:
            # Catch errors such as corrupted/unsupported checkpoint files and fall back
            print(f"Failed to load YOLO model from '{YOLO_MODEL_PATH}': {e}")
            _yolo_model = None
    return _yolo_model
