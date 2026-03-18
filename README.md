# DeepShield — Deepfake Detection System

Detects deepfakes in images and audio using two trained models:
- **Image**: EfficientNetB0 (transfer learning, ImageNet weights)
- **Voice**: CNN + MFCC feature extraction

---

## Language Composition

This project is built with:
- **HTML**: 90.9%
- **Python**: 9.1%

---

## Project Structure

```
deepshield/
├── main.py                     # FastAPI server
├── image_detector.py           # EfficientNetB0 inference
├── voice_detector.py           # CNN+MFCC inference
├── train_deepfake_model.py     # Image model training
├── training.py                 # Voice model training
├── requirements.txt
├── saved_models/               # Put your .h5 weights here
│   ├── deepfake_detector_3.h5
│   └── deepfake_voice_detection.h5
└── frontend/
    └── index.html
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Train the models

**Image model:**
```bash
python train_deepfake_model.py
```
Weights saved to `deepfake_detector_3.h5` — move to `saved_models/`

**Voice model:**
```bash
python training.py
```
Weights saved to `deepfake_voice_detection.h5` — move to `saved_models/`

---

## Run

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --no-reload
```

Open **http://localhost:8000** in your browser.

---

## API

| Method | Endpoint          | Input       | Returns                          |
|--------|-------------------|-------------|----------------------------------|
| POST   | `/analyze/image`  | image file  | verdict, fake %, face count      |
| POST   | `/analyze/voice`  | audio file  | verdict, fake %, duration        |
| GET    | `/health`         | —           | `{"status": "running"}`      |

---

## Repository Information

- **Repository**: rithikagirish/DeepShield
- **Repository ID**: 1185625485
- **Last Updated**: 2026-03-18