import io
import time
import pickle
import numpy as np
from pathlib import Path

MODEL_PATH  = "saved_models/deepfake_voice_detection.h5"
SCALER_PATH = "saved_models/deepfake_voice_detection_scaler.pkl"
THRESHOLD   = 0.70
SAMPLE_RATE = 22050
N_MFCC      = 40
MIN_DURATION = 3.0

_model  = None
_scaler = None


def load_model():
    global _model, _scaler
    import tensorflow as tf

    if Path(MODEL_PATH).exists():
        print(f"Loading voice model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Voice model loaded.")
    else:
        print(f"Warning: No weights found at {MODEL_PATH}. Model not loaded.")

    if Path(SCALER_PATH).exists():
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        print("Scaler loaded.")


def _extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    chroma      = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel         = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db      = librosa.power_to_db(mel, ref=np.max)
    sc          = librosa.feature.spectral_contrast(y=audio, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(audio).mean()
    rms         = librosa.feature.rms(y=audio).mean()
    bw          = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()
    centroid    = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()

    features = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
        mel_db.mean(axis=1)[:7], mel_db.std(axis=1)[:7],
        sc.mean(axis=1),
        [zcr, rms, bw, centroid],
    ])
    return features.astype(np.float32)


def predict_voice(audio_bytes: bytes) -> dict:
    import librosa

    if _model is None:
        load_model()

    if _model is None:
        return {
            "verdict": "error",
            "message": "Model weights not found. Train the model first.",
            "fake_probability": 0.0,
            "real_probability": 0.0,
        }

    # Load audio
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    duration  = len(audio) / sr

    if duration < MIN_DURATION:
        raise ValueError(f"Audio too short ({duration:.1f}s). Minimum is {MIN_DURATION}s.")

    # Extract features
    t0 = time.perf_counter()
    features = _extract_features(audio, sr)

    if _scaler is not None:
        features = _scaler.transform(features.reshape(1, -1)).flatten()

    # Reshape for Conv1D: (1, n_features, 1)
    inp   = features.reshape(1, -1, 1)
    score = float(_model.predict(inp, verbose=0)[0][0])
    ms    = round((time.perf_counter() - t0) * 1000, 1)

    is_fake = score > THRESHOLD

    return {
        "verdict":          "fake" if is_fake else "real",
        "fake_probability": round(score * 100, 2),
        "real_probability": round((1 - score) * 100, 2),
        "confidence":       round(max(score, 1 - score) * 100, 2),
        "duration_sec":     round(duration, 2),
        "inference_ms":     ms,
        "model":            "CNN+MFCC",
    }
