# main.py
import io
import os
from typing import Dict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from PIL import Image
import numpy as np

APP_DETECTOR = "retinaface"
DEFAULT_THRESHOLD = float(os.getenv("ARC_THRESHOLD", "0.38"))

app = FastAPI(title="Face Verify API", version="1.0.0")

# CORS opcional (útil para pruebas desde frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pil_to_array(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}

@app.post("/verify")
async def verify(
    id_image: UploadFile = File(..., description="Imagen del documento (frontal)"),
    selfie: UploadFile = File(..., description="Selfie a validar"),
    threshold: float = Form(DEFAULT_THRESHOLD)
) -> Dict:
    try:
        id_bytes = await id_image.read()
        sf_bytes = await selfie.read()

        id_pil = Image.open(io.BytesIO(id_bytes))
        sf_pil = Image.open(io.BytesIO(sf_bytes))

        id_arr = pil_to_array(id_pil)
        sf_arr = pil_to_array(sf_pil)

        result = DeepFace.verify(
            img1_path=id_arr,
            img2_path=sf_arr,
            model_name="ArcFace",
            detector_backend=APP_DETECTOR,
            distance_metric="cosine",
            enforce_detection=True
        )
        dist = float(result.get("distance", 1.0))
        verified = dist <= float(threshold)

        return {
            "ok": True,
            "verified": bool(verified),
            "distance": dist,
            "threshold": float(threshold),
            "model": "ArcFace",
            "detector": APP_DETECTOR
        }
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": f"{type(e).__name__}: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
