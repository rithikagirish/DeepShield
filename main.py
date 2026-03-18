from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from image_detector import predict_image
from voice_detector import predict_voice

app = FastAPI(title="DeepShield", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    result = predict_image(image_bytes)
    result["filename"] = file.filename
    return result


@app.post("/analyze/voice")
async def analyze_voice(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    audio_bytes = await file.read()

    try:
        result = predict_voice(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result["filename"] = file.filename
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
