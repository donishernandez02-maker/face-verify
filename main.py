# main.py
import io
import os
from typing import Dict, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from deepface import DeepFace
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2

# =========================
# Config
# =========================
APP_DETECTOR_DEFAULT = "auto"       # auto | retinaface | opencv | mtcnn | ssd | yolov8 | mediapipe ...
DEFAULT_THRESHOLD     = float(os.getenv("ARC_THRESHOLD", "0.38"))

# Heurísticas de documento (ID card)
CARD_MIN_AREA_FRAC    = 0.20   # el cuadrilátero más grande debe cubrir >=20% del frame
CARD_MAX_AREA_FRAC    = 0.95   # y no más del 95%
CARD_AR_MIN           = 1.20   # ratio ancho/alto esperado (IDs ~1.4–1.7, margen para cédulas)
CARD_AR_MAX           = 1.95
CARD_ANGLE_TOL        = 25.0   # tolerancia de ortogonalidad (grados)

# Selfie / Face
MIN_FACE_FRAC         = 0.12   # cara debe ocupar al menos 12% del menor lado (aprox)
MAX_SELFIE_FACES      = 1      # exactamente una cara en selfie

app = FastAPI(title="Face Verify API", version="1.2.0")

# CORS (útil para frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Utils
# =========================
def pil_to_array(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

def upscale_if_tiny(pil_img: Image.Image, min_side_target: int = 400) -> Image.Image:
    w, h = pil_img.size
    m = min(w, h)
    if m < min_side_target:
        scale = float(min_side_target) / float(m)
        new_size = (int(w * scale), int(h * scale))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
    return pil_img

def _angle_cos(p0, p1, p2) -> float:
    # cos del ángulo en p1 (entre p0->p1 y p2->p1)
    d1 = p0 - p1
    d2 = p2 - p1
    cosang = abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9))
    return cosang

def _quad_is_rect_like(cnt: np.ndarray) -> bool:
    # Aproxima si los 4 vértices forman un rectángulo (ángulos ~90°)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    # ordena cuadrilátero por perímetro mínimo (approx)
    rect = cv2.convexHull(pts)
    if len(rect) != 4:
        rect = pts
    if len(rect) != 4:
        return False
    # evaluar ortogonalidad con cosenos -> 0 ~ 90°
    rect = rect.reshape(4, 2)
    angles = []
    for i in range(4):
        p0 = rect[(i - 1) % 4]
        p1 = rect[i]
        p2 = rect[(i + 1) % 4]
        cosang = _angle_cos(p0, p1, p2)  # 0 ideal
        ang = np.degrees(np.arccos(max(min(1.0 - 1e-6, 1.0 - cosang + 1e-6), -1.0 + 1e-6)))  # heurístico estable
        # En la práctica, usamos tolerancia grande:
        if abs(90 - ang) > CARD_ANGLE_TOL:
            return False
        angles.append(ang)
    return True

def detect_card_quad(bgr: np.ndarray) -> Tuple[bool, Dict]:
    """Detecta si hay un cuadrilátero grande tipo tarjeta/ID."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, {"reason": "no_contours"}

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    area_img = w * h

    for c in cnts[:10]:
        area = cv2.contourArea(c)
        if area <= 1:
            continue
        frac = area / (area_img + 1e-9)
        if frac < CARD_MIN_AREA_FRAC or frac > CARD_MAX_AREA_FRAC:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # Análisis de aspect ratio
            x, y, ww, hh = cv2.boundingRect(approx)
            ar = ww / float(hh + 1e-9)
            rect_like = _quad_is_rect_like(approx)
            if (CARD_AR_MIN <= ar <= CARD_AR_MAX) and rect_like:
                return True, {"area_frac": frac, "ar": ar, "rect_like": True}
    return False, {"reason": "no_quad_like_card"}

def extract_faces(img_rgb: np.ndarray, backend: str) -> List[dict]:
    """
    DeepFace.extract_faces devuelve lista de dicts: { "face": np.ndarray, "facial_area": {...}, "confidence": ... }
    """
    faces = DeepFace.extract_faces(img_path=img_rgb, detector_backend=backend, enforce_detection=False)
    return faces

def pick_best_face(faces: List[dict], prefer_larger: bool = True) -> np.ndarray:
    """Escoge la cara más grande o de mayor confianza."""
    if not faces:
        return None
    if prefer_larger:
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0)),
            reverse=True
        )
    else:
        faces_sorted = sorted(faces, key=lambda f: f.get("confidence", 0.0), reverse=True)
    face_img = faces_sorted[0].get("face")
    if isinstance(face_img, np.ndarray):
        return face_img
    # DeepFace a veces trae PIL; normalizamos
    arr = faces_sorted[0].get("face")
    if hasattr(arr, "shape"):
        return np.array(arr)
    return None

def do_verify(id_arr, sf_arr, threshold: float, detector: str, enforce: bool) -> Tuple[dict, dict]:
    """Ejecuta DeepFace.verify con parámetros dados. Devuelve (payload, meta)."""
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
    payload = {
        "ok": True,
        "verified": bool(verified),
        "distance": dist,
        "threshold": float(threshold),
        "model": "ArcFace",
        "detector": detector,
        "enforce": bool(enforce),
    }
    meta = {"fallback_used": False}
    return payload, meta

# =========================
# Endpoints
# =========================
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}

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

@app.post("/verify")
async def verify(
    id_image: UploadFile = File(..., description="Imagen del documento (frontal)"),
    selfie:   UploadFile = File(..., description="Selfie a validar"),
    threshold: float = Form(DEFAULT_THRESHOLD),
    detector:  str   = Form(APP_DETECTOR_DEFAULT),  # auto (default) | retinaface | opencv | ...
    enforce:   bool  = Form(True)                   # aplica si detector != auto
) -> Dict:
    try:
        # --------------------------
        # 1) Cargar imágenes
        # --------------------------
        id_bytes = await id_image.read()
        sf_bytes = await selfie.read()

        id_pil = Image.open(io.BytesIO(id_bytes))
        sf_pil = Image.open(io.BytesIO(sf_bytes))

        id_pil = upscale_if_tiny(id_pil, 500)
        sf_pil = upscale_if_tiny(sf_pil, 500)

        id_arr_rgb = pil_to_array(id_pil)
        sf_arr_rgb = pil_to_array(sf_pil)

        # BGR para OpenCV (sólo para heurísticas del documento)
        id_bgr = cv2.cvtColor(id_arr_rgb, cv2.COLOR_RGB2BGR)

        # --------------------------
        # 2) Validar DOCUMENTO (heurística + cara pequeña en documento)
        # --------------------------
        card_like, card_meta = detect_card_quad(id_bgr)

        # Detectar caras en doc con backend robusto (fallback si hace falta)
        doc_faces = []
        try:
            doc_faces = extract_faces(id_arr_rgb, backend="retinaface")
        except Exception:
            try:
                doc_faces = extract_faces(id_arr_rgb, backend="opencv")
            except Exception:
                doc_faces = []

        # cara “pequeña” en doc (respecto al frame)
        doc_h, doc_w = id_arr_rgb.shape[:2]
        doc_ok_face = False
        if doc_faces:
            f = pick_best_face(doc_faces, prefer_larger=False)  # en ID suele ser rostro pequeño
            if f is not None:
                # estimar tamaño en base al área de facial_area si está
                area_ok = False
                try:
                    fa = sorted(doc_faces, key=lambda d: d.get("confidence", 0), reverse=True)[0].get("facial_area", {})
                    fw, fh = fa.get("w", 0), fa.get("h", 0)
                    min_side = min(doc_w, doc_h)
                    # rostro “pequeño” => menor que 40% del lado mínimo del ID (heurística)
                    area_ok = max(fw, fh) <= 0.40 * float(min_side)
                except Exception:
                    area_ok = True  # si no hay datos, no bloqueamos

                doc_ok_face = area_ok

        doc_ok = bool(card_like and doc_ok_face)

        # --------------------------
        # 3) Validar SELFIE (exactamente 1 rostro, suficientemente grande)
        # --------------------------
        sf_faces = []
        try:
            sf_faces = extract_faces(sf_arr_rgb, backend="retinaface")
        except Exception:
            try:
                sf_faces = extract_faces(sf_arr_rgb, backend="opencv")
            except Exception:
                sf_faces = []

        selfie_ok = False
        selfie_face_img = None
        if sf_faces and len(sf_faces) == MAX_SELFIE_FACES:
            # cara principal
            face_pick = pick_best_face(sf_faces, prefer_larger=True)
            if face_pick is not None:
                selfie_face_img = face_pick
                # comprobar tamaño de rostro
                try:
                    fa = sf_faces[0].get("facial_area", {})
                    fw, fh = fa.get("w", 0), fa.get("h", 0)
                except Exception:
                    fw = fh = 0
                sh, sw = sf_arr_rgb.shape[:2]
                min_side = float(min(sw, sh))
                selfie_ok = max(fw, fh) >= MIN_FACE_FRAC * min_side

        # Si no cumple, rechazamos con razón clara
        if not doc_ok or not selfie_ok:
            return {
                "ok": False,
                "doc_ok": bool(doc_ok),
                "selfie_ok": bool(selfie_ok),
                "reason": {
                    "doc": ("not_card_like" if not card_like else
                            "no_small_face_on_id" if not doc_ok_face else "ok"),
                    "selfie": ("face_count!=1_or_too_small" if not selfie_ok else "ok")
                }
            }

        # --------------------------
        # 4) Comparación biométrica (match)
        #    - Recortamos rostro del doc y rostro de la selfie
        # --------------------------
        # mejor cara del documento (usamos la de mayor confianza)
        doc_face_img = pick_best_face(doc_faces, prefer_larger=False)
        if doc_face_img is None or selfie_face_img is None:
            return {"ok": False, "reason": "face_crop_failed"}

        # DeepFace.verify acepta arrays RGB
        id_face_arr = np.array(doc_face_img)
        sf_face_arr = np.array(selfie_face_img)

        # 4.a) modo auto (retinaface + fallback a opencv)
        def verify_auto(x1, x2, thr):
            try:
                payload, _ = do_verify(x1, x2, thr, detector="retinaface", enforce=True)
                payload["fallback_used"] = False
                payload["detector"] = "retinaface"
                payload["enforce"] = True
                return payload
            except Exception as e1:
                try:
                    payload, _ = do_verify(x1, x2, thr, detector="opencv", enforce=False)
                    payload["fallback_used"] = True
                    payload["detector"] = "opencv"
                    payload["enforce"] = False
                    payload["note"] = f"retinaface failed: {type(e1).__name__}"
                    return payload
                except Exception as e2:
                    return {"ok": False, "error": f"FallbackError: {type(e1).__name__} -> {type(e2).__name__}"}

        if detector.lower() == "auto":
            ver = verify_auto(id_face_arr, sf_face_arr, threshold)
        else:
            # modo manual
            try:
                ver, _ = do_verify(id_face_arr, sf_face_arr, threshold, detector=detector, enforce=enforce)
                ver["fallback_used"] = False
            except Exception as e:
                ver = {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

        # empaquetar estado de validaciones previas
        ver["doc_ok"] = True
        ver["selfie_ok"] = True
        return ver

    except Exception as e:
        # no tiramos 500 para facilitar manejo en el cliente
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": f"{type(e).__name__}: {str(e)}"}
        )

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
