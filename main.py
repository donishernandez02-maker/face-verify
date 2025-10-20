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
APP_VERSION           = "1.5.0"
APP_DETECTOR_DEFAULT  = "auto"       # auto | retinaface | opencv | mtcnn | ssd | yolov8 | mediapipe ...
DEFAULT_THRESHOLD     = float(os.getenv("ARC_THRESHOLD", "0.38"))
DOC_MODE_DEFAULT      = "auto"       # auto (strict->loose) | strict | loose

# Heurísticas de documento (ID card) — relajadas para robustez
CARD_MIN_AREA_FRAC    = 0.08
CARD_MAX_AREA_FRAC    = 0.98
CARD_AR_MIN           = 0.90
CARD_AR_MAX           = 2.20
CARD_ANGLE_TOL        = 35.0

# Selfie / Face
MIN_FACE_FRAC         = 0.12   # cara ≥12% del lado menor del frame
MAX_SELFIE_FACES      = 1      # exactamente una cara en selfie

app = FastAPI(title="Face Verify API", version=APP_VERSION)

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
    """Corrige rotación según EXIF (verticales)."""
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
    d1 = p0 - p1
    d2 = p2 - p1
    return abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9))

def _quad_is_rect_like(cnt: np.ndarray) -> bool:
    """¿Los 4 vértices forman un rectángulo (∼90° ± tolerancia)?"""
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
        cosang = _angle_cos(p0, p1, p2)
        ang = np.degrees(np.arccos(max(min(1.0 - 1e-6, 1.0 - cosang + 1e-6), -1.0 + 1e-6)))
        if abs(90 - ang) > CARD_ANGLE_TOL:
            return False
    return True

def detect_card_quad(bgr: np.ndarray) -> Tuple[bool, Dict]:
    """Detecta cuadrilátero grande tipo tarjeta/ID en la orientación dada."""
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
    """Prueba la heurística tarjeta a 0/90/180/270°. Devuelve (ok, meta) con meta['rotation']."""
    candidates = []
    for k in (0, 1, 2, 3):
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
    """DeepFace.extract_faces → lista de dicts."""
    return DeepFace.extract_faces(img_path=img_rgb, detector_backend=backend, enforce_detection=False)

def pick_best_face(faces: List[dict], prefer_larger: bool = True) -> np.ndarray:
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
    """DeepFace.verify con parámetros dados."""
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
    return {"ok": True, "status": "healthy", "version": APP_VERSION}

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
            info["id_image"]["size"]   = id_pil.size
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
    enforce:   bool  = Form(True),
    doc_mode:  str   = Form(DOC_MODE_DEFAULT)       # strict | loose | auto
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

        # cara “pequeña” en doc
        doc_h, doc_w = id_arr_rgb.shape[:2]
        doc_ok_face = False
        if doc_faces:
            f = pick_best_face(doc_faces, prefer_larger=False)  # retrato impreso suele ser pequeño
            if f is not None:
                try:
                    fa = sorted(doc_faces, key=lambda d: d.get("confidence", 0), reverse=True)[0].get("facial_area", {})
                    fw, fh = fa.get("w", 0), fa.get("h", 0)
                    min_side = min(doc_w, doc_h)
                    doc_ok_face = max(fw, fh) <= 0.40 * float(min_side)
                except Exception:
                    doc_ok_face = True  # si no hay datos, no bloqueamos

        # ---- Decisión strict / loose / auto ----
        doc_mode_req = (doc_mode or DOC_MODE_DEFAULT).lower().strip()
        doc_mode_used = "strict"  # default inicial

        def strict_decision() -> bool:
            return bool(card_like and doc_ok_face)

        def loose_decision() -> Tuple[bool, Dict]:
            # Señales globales: AR + densidad de bordes
            bgr_full = cv2.cvtColor(id_arr_rgb, cv2.COLOR_RGB2BGR)
            ar_global = doc_w / float(doc_h + 1e-9)
            dens = edge_density(bgr_full)
            ar_ok   = 0.80 <= ar_global <= 2.50
            dens_ok = 0.008 <= dens <= 0.30
            ok = bool(doc_ok_face and ar_ok and dens_ok)
            meta_patch = {
                "loose": True,
                "ar_global": ar_global,
                "edge_density": dens
            }
            return ok, meta_patch

        if doc_mode_req == "strict":
            doc_ok = strict_decision()
            doc_mode_used = "strict"
        elif doc_mode_req == "loose":
            doc_ok, meta_patch = loose_decision()
            doc_mode_used = "loose"
            if doc_ok:
                card_meta = {**(card_meta or {}), **meta_patch}
        else:  # auto (strict → loose)
            doc_ok = strict_decision()
            doc_mode_used = "strict"
            if not doc_ok:
                doc_ok, meta_patch = loose_decision()
                if doc_ok:
                    doc_mode_used = "loose"
                    card_meta = {**(card_meta or {}), **meta_patch}

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

        # Si falla doc o selfie, responder claro
        if not doc_ok or not selfie_ok:
            return {
                "ok": False,
                "doc_ok": bool(doc_ok),
                "selfie_ok": bool(selfie_ok),
                "reason": {
                    "doc": ("not_card_like" if not doc_ok else "ok"),
                    "selfie": ("face_count!=1_or_too_small" if not selfie_ok else "ok")
                },
                "card_meta": card_meta,
                "doc_mode_used": doc_mode_used,
                "version": APP_VERSION
            }

        # --------------------------
        # 4) Comparación biométrica (match)
        # --------------------------
        doc_face_img = pick_best_face(doc_faces, prefer_larger=False)
        if doc_face_img is None or selfie_face_img is None:
            return {"ok": False, "reason": "face_crop_failed", "doc_mode_used": doc_mode_used, "version": APP_VERSION}

        id_face_arr = np.array(doc_face_img)
        sf_face_arr = np.array(selfie_face_img)

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

        if (detector or "auto").lower() == "auto":
            ver = verify_auto(id_face_arr, sf_face_arr, threshold)
        else:
            try:
                ver, _ = do_verify(id_face_arr, sf_face_arr, threshold, detector=detector, enforce=enforce)
                ver["fallback_used"] = False
            except Exception as e:
                ver = {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}

        ver["doc_ok"] = True
        ver["selfie_ok"] = True
        ver["card_meta"] = card_meta
        ver["doc_mode_used"] = doc_mode_used
        ver["version"] = APP_VERSION
        return ver

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": f"{type(e).__name__}: {str(e)}", "version": APP_VERSION}
        )

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
