# main.py
import io
import os
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from deepface import DeepFace
from PIL import Image, UnidentifiedImageError
import numpy as np

# Config
APP_DETECTOR = "retinaface"  # retinaface | opencv | mtcnn | ssd | yolov8 | mediapipe ...
DEFAULT_THRESHOLD = float(os.getenv("ARC_THRESHOLD", "0.38"))

app = FastAPI(title="Face Verify API", version="1.0.0")

# CORS (útil para pruebas desde frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utils
def pil_to_array(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

def upscale_if_tiny(pil_img: Image.Image, min_side_target: int = 400) -> Image.Image:
    """Si la imagen es muy pequeña, la escala para dar más señal al detector/embeddings."""
    w, h = pil_img.size
    m = min(w, h)
    if m < min_side_target:
        scale = float(min_side_target) / float(m)
        new_size = (int(w * scale), int(h * scale))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
    return pil_img

# Health
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}

# Debug de uploads
@app.post("/debug")
async def debug_uploads(
    id_image: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    try:
        id_bytes = await id_image.read()
        sf_bytes = await selfie.read()

        info = {"ok": True, "id_image": {}, "selfie": {}}

        # ID
        info["id_image"]["bytes"] = len(id_bytes)
        try:
            id_pil = Image.open(io.BytesIO(id_bytes))
            info["id_image"]["format"] = id_pil.format
            info["id_image"]["size"]   = id_pil.size  # (w, h)
            info["id_image"]["mode"]   = id_pil.mode
        except UnidentifiedImageError as e:
            info["id_image"]["error"] = f"PIL: {str(e)}"

        # Selfie
        info["selfie"]["bytes"] = len(sf_bytes)
        try:
            sf_pil = Image.open(io.BytesIO(sf_bytes))
            info["selfie"]["format"] = sf_pil.format
            info["selfie"]["size"]   = sf_pil.size
            info["selfie"]["mode"]   = sf_pil.mode
        except UnidentifiedImageError as e:
            info["selfie"]["error"] = f"PIL: {str(e)}"

        return info
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

# Verificación
@app.post("/verify")
async def verify(
    id_image: UploadFile = File(..., description="Imagen del documento (frontal)"),
    selfie:   UploadFile = File(..., description="Selfie a validar"),
    threshold: float = Form(DEFAULT_THRESHOLD),
    detector:  str   = Form(APP_DETECTOR),  # permite sobreescribir: opencv, mtcnn, etc.
    enforce:   bool  = Form(True)           # exigir detección de rostro (True recomendado)
) -> Dict:
    try:
        # Leer archivos
        id_bytes = await id_image.read()
        sf_bytes = await selfie.read()

        # Abrir con PIL
        id_pil = Image.open(io.BytesIO(id_bytes))
        sf_pil = Image.open(io.BytesIO(sf_bytes))

        # Reescalar si llegan muy pequeñas
        id_pil = upscale_if_tiny(id_pil, 400)
        sf_pil = upscale_if_tiny(sf_pil, 400)

        # A array RGB
        id_arr = pil_to_array(id_pil)
        sf_arr = pil_to_array(sf_pil)

        # Verificar con DeepFace (ArcFace + métrica coseno)
        result = DeepFace.verify(
            img1_path=id_arr,
            img2_path=sf_arr,
            model_name="ArcFace",
            detector_backend=detector,
            distance_metric="cosine",
            enforce_detection=bool(enforce)
        )

        dist = float(result.get("distance", 1.0))
        verified = dist <= float(threshold)

        return {
            "ok": True,
            "verified": bool(verified),
            "distance": dist,
            "threshold": float(threshold),
            "model": "ArcFace",
            "detector": detector,
            "enforce": bool(enforce)
        }

    except Exception as e:
        # No tiramos 500 para facilitar el manejo del cliente
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": f"{type(e).__name__}: {str(e)}"}
        )

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
