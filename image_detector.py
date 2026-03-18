import time
import numpy as np
import cv2
from pathlib import Path

MODEL_PATH = "saved_models/deepfake_detector_3.h5"
IMAGE_SIZE = (224, 224)
THRESHOLD  = 0.65

_model = None


def load_model():
    global _model
    import tensorflow as tf

    if Path(MODEL_PATH).exists():
        print(f"Loading image model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Image model loaded.")
    else:
        print(f"Warning: No weights found at {MODEL_PATH}. Model not loaded.")


def _preprocess(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image.")

    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def _detect_faces(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return []

    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in faces]


def predict_image(image_bytes: bytes) -> dict:
    if _model is None:
        load_model()

    if _model is None:
        return {
            "verdict": "error",
            "message": "Model weights not found. Train the model first.",
            "fake_probability": 0.0,
            "real_probability": 0.0,
        }

    t0 = time.perf_counter()
    inp = _preprocess(image_bytes)
    score = float(_model.predict(inp, verbose=0)[0][0])
    ms = round((time.perf_counter() - t0) * 1000, 1)

    is_fake = score > THRESHOLD
    faces   = _detect_faces(image_bytes)

    return {
        "verdict":          "fake" if is_fake else "real",
        "fake_probability": round(score * 100, 2),
        "real_probability": round((1 - score) * 100, 2),
        "confidence":       round(max(score, 1 - score) * 100, 2),
        "face_count":       len(faces),
        "faces":            faces,
        "inference_ms":     ms,
        "model":            "EfficientNetB0",
    }
