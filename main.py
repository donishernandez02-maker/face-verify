# main.py
import io
import os
from typing import Dict, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from deepface import DeepFace
from PIL import Image, UnidentifiedImageError, ImageOps
import numpy as np
import cv2

# =========================
# Config
# =========================
APP_DETECTOR_DEFAULT = "auto"       # auto | retinaface | opencv | mtcnn | ssd | yolov8 | mediapipe ...
DEFAULT_THRESHOLD     = float(os.getenv("ARC_THRESHOLD", "0.38"))

# Heurísticas de documento (ID card) — relajadas para robustez
CARD_MIN_AREA_FRAC    = 0.08   # antes 0.20
CARD_MAX_AREA_FRAC    = 0.98   # antes 0.95
CARD_AR_MIN           = 0.90   # antes 1.20
CARD_AR_MAX           = 2.20   # antes 1.95
CARD_ANGLE_TOL        = 35.0   # antes 25.0

# Selfie / Face
MIN_FACE_FRAC         = 0.12   # cara debe ocupar al menos 12% del lado menor del frame
MAX_SELFIE_FACES      = 1      # exactamente una cara en selfie

app = FastAPI(title="Face Verify API", version="1.3.0")

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

def auto_orient(pil_img: Image.Image) -> Image.Image:
    """Corrige rotación según EXIF (fotos verticales)."""
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img

def upscale_if_tiny(pil_img: Image.Image, min_side_target: int = 500) -> Image.Image:
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
    """Aproxima si los 4 vértices forman un rectángulo (ángulos ~90° con tolerancia)."""
    pts = cnt.reshape(-1, 2).astype(np.float32)
    rect = cv2.convexHull(pts)
    if len(rect) != 4:
        rect = pts
    if len(rect) != 4:
        return False
    rect = rect.reshape(4, 2)
    for i in range(4):
        p0 = rect[(i - 1) % 4]
        p1 = rect[i]
        p2 = rect[(i + 1) % 4]
        cosang = _angle_cos(p0, p1, p2)  # 0 ideal
        # convertir a grados (heurístico estable)
        ang = np.degrees(np.arccos(max(min(1.0 - 1e-6, 1.0 - cosang + 1e-6), -1.0 + 1e-6)))
        if abs(90 - ang) > CARD_ANGLE_TOL:
            return False
    return True

def detect_card_quad(bgr: np.ndarray) -> Tuple[bool, Dict]:
    """Detecta si hay un cuadrilátero grande tipo tarjeta/ID en orientación dada."""
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
            x, y, ww, hh = cv2.boundingRect(approx)
            ar = ww / float(hh + 1e-9)
            rect_like = _quad_is_rect_like(approx)
            if (CARD_AR_MIN <= ar <= CARD_AR_MAX) and rect_like:
                return True, {"area_frac": frac, "ar": ar, "rect_like": True}
    return False, {"reason": "no_quad_like_card"}

def edge_density(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, 50, 150)
    h, w = e.shape[:2]
    return float(np.count_nonzero(e)) / float(h * w + 1e-9)

def detect_card_any_rotation(img_rgb: np.ndarray):
    """
    Prueba la heurística de tarjeta a 0°, 90°, 180°, 270°.
    Devuelve (ok, meta) y meta['rotation'] en grados.
    """
    candidates = []
    for k in (0, 1, 2, 3):  # rotaciones 0/90/180/270
        test_rgb = np.rot90(img_rgb, k=k) if k else img_rgb
        bgr = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2BGR)
        ok, meta = detect_card_quad(bgr)
        meta = meta or {}
        meta['rotation'] = k * 90
        if ok:
            return True, meta
        candidates.append((meta.get('area_frac', 0.0), meta))
    best = max(candidates, key=lambda x: x[0])[1] if candidates else {"rotation": 0}
    return False, best

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
    enforce:   bool  = Form(True),                  # aplica si detector != auto
    doc_mode:  str   = Form("strict")               # 'strict' | 'loose'
) -> Dict:
    try:
        # --------------------------
        # 1) Cargar imágenes + auto-orientación + upscale
        # --------------------------
        id_bytes = await id_image.read()
        sf_bytes = await selfie.read()

        id_pil = auto_orient(Image.open(io.BytesIO(id_bytes)))
        sf_pil = auto_orient(Image.open(io.BytesIO(sf_bytes)))

        id_pil = upscale_if_tiny(id_pil, 500)
        sf_pil = upscale_if_tiny(sf_pil, 500)

        id_arr_rgb = pil_to_array(id_pil)
        sf_arr_rgb = pil_to_array(sf_pil)

        # --------------------------
        # 2) Validar DOCUMENTO (rotation-aware + heurística de tarjeta + rostro pequeño)
        # --------------------------
        # Intento con rotaciones para admitir fotos en vertical
        card_like, card_meta = detect_card_any_rotation(id_arr_rgb)

        # Detectar caras en doc
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
            f = pick_best_face(doc_faces, prefer_larger=False)  # retrato impreso suele ser pequeño
            if f is not None:
                try:
                    fa = sorted(doc_faces, key=lambda d: d.get("confidence", 0), reverse=True)[0].get("facial_area", {})
                    fw, fh = fa.get("w", 0), fa.get("h", 0)
                    min_side = min(doc_w, doc_h)
                    # pequeño => <= 40% del lado mínimo (heurística)
                    doc_ok_face = max(fw, fh) <= 0.40 * float(min_side)
                except Exception:
                    doc_ok_face = True  # si no hay datos, no bloqueamos

        # Decisión estricta
        doc_ok = bool(card_like and doc_ok_face)

        # --- Modo laxo (por si la foto está vertical o con fondo complejo) ---
        if not doc_ok and doc_mode.lower() == "loose":
            # Señales alternativas: rostro pequeño presente + AR global razonable + densidad de bordes en rango
            bgr_full = cv2.cvtColor(id_arr_rgb, cv2.COLOR_RGB2BGR)
            ar_global = doc_w / float(doc_h + 1e-9)
            dens = edge_density(bgr_full)
            # rangos amplios pensados para ID en vertical/horizontal y escenas reales
            ar_ok   = 0.80 <= ar_global <= 2.50
            dens_ok = 0.008 <= dens <= 0.30
            if doc_ok_face and ar_ok and dens_ok:
                doc_ok = True
                card_meta = {
                    "rotation": card_meta.get("rotation", 0),
                    "loose": True,
                    "ar_global": ar_global,
                    "edge_density": dens
                }

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
            face_pick = pick_best_face(sf_faces, prefer_larger=True)
            if face_pick is not None:
                selfie_face_img = face_pick
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
                    "doc": ("not_card_like" if not doc_ok else "ok"),
                    "selfie": ("face_count!=1_or_too_small" if not selfie_ok else "ok")
                },
                "card_meta": card_meta
            }

        # --------------------------
        # 4) Comparación biométrica (match)
        #    - Recortamos rostro del doc y rostro de la selfie
        # --------------------------
        doc_face_img = pick_best_face(doc_faces, prefer_larger=False)
        if doc_face_img is None or selfie_face_img is None:
            return {"ok": False, "reason": "face_crop_failed"}

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
            try:
                ver, _ = do_verify(id_face_arr, sf_face_arr, threshold, detector=detector, enforce=enforce)
                ver["fallback_used"] = False
            except Exception as e:
                ver = {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

        # empaquetar estado de validaciones previas
        ver["doc_ok"] = True
        ver["selfie_ok"] = True
        ver["card_meta"] = card_meta
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
