# main.py  — Face Verify API (v1.7.0)
import io
import os
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ExifTags
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# CV / Biometrics
import cv2
from deepface import DeepFace

# =========================
# Config
# =========================
APP_DETECTOR = os.getenv("DF_DETECTOR", "retinaface")  # retinaface|opencv|mtcnn...
DEFAULT_THRESHOLD = float(os.getenv("ARC_THRESHOLD", "0.38"))
VERSION = "1.7.0"

# Heurísticos documento
AR_ID = 1.58  # ancho/alto de tarjeta ID estandar
AR_TOL = 0.55  # tolerancia relajada (no importa si ignore_ar=True)
EDGE_MAX_GLOBAL_DEF = 0.55
TEXT_MAX_CENTER_DEF = 0.18
EDGE_MIN = 0.01
TEXT_MIN = 0.01

# Heurísticos selfie (Haar cascades)
FACE_MIN_AREA_RATIO = 0.18   # rostro debe cubrir >=18% del área del frame
EYE_MIN_COUNT = 2            # al menos 2 ojos encontrados
BRIGHT_MIN = 30              # brillo mínimo
BRIGHT_MAX = 230             # brillo máximo

# =========================
# App
# =========================
app = FastAPI(title="Face Verify API", version=VERSION)

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
def _fix_exif_orientation(pil_img: Image.Image) -> Image.Image:
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    return pil_img


def load_as_rgb_array(raw: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(raw))
    pil = _fix_exif_orientation(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return np.array(pil)


def center_crop(img: np.ndarray, frac: float = 0.6) -> np.ndarray:
    h, w = img.shape[:2]
    cw, ch = int(w * frac), int(h * frac)
    x1 = (w - cw) // 2
    y1 = (h - ch) // 2
    return img[y1:y1 + ch, x1:x1 + cw]


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float((edges > 0).mean())


def text_like_density(gray: np.ndarray) -> float:
    # proxy para texto: blackhat (resalta letras oscuras sobre fondo claro) + binarización
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((bw > 0).mean())


def aspect_ratio_ok(rot_rect: Tuple[Tuple[float, float], Tuple[float, float], float],
                    ignore_ar: bool) -> bool:
    (_, _), (w, h), _ = rot_rect
    if w == 0 or h == 0:
        return False
    ar = max(w, h) / max(1.0, min(w, h))
    if ignore_ar:
        return True
    # Debe ser "tarjeta": ~1.58 (+/- tolerancia)
    return abs(ar - AR_ID) <= AR_TOL


def largest_card_like_quad(bw: np.ndarray) -> Optional[Tuple]:
    # Encuentra el rectángulo rotado del contorno más grande
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect = cv2.minAreaRect(contours[0])
    return rect


def doc_quadrilateral_check(rgb: np.ndarray,
                            ignore_ar: bool,
                            require_quad: bool) -> Tuple[bool, dict]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    ret, bw = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    rect = largest_card_like_quad(bw)
    if rect is None:
        return (False if require_quad else True, {"reason": "no_quad_like_card", "rotation": 0})

    ok_ar = aspect_ratio_ok(rect, ignore_ar)
    if not ok_ar and require_quad:
        return False, {"reason": "bad_card_ar", "rotation": rect[2]}

    return True, {"reason": "ok", "rotation": rect[2]}


def doc_metrics(rgb: np.ndarray,
                use_center: bool) -> Tuple[dict, dict]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g_ed = edge_density(gray)
    g_tx = text_like_density(gray)
    h, w = gray.shape[:2]
    g_ar = w / max(1, h)

    metrics_global = {
        "ar_global": float(g_ar),
        "edge_density": float(g_ed),
        "text_density": float(g_tx),
    }

    if use_center:
        ctr = center_crop(rgb, 0.6)
        cgray = cv2.cvtColor(ctr, cv2.COLOR_RGB2GRAY)
        c_ed = edge_density(cgray)
        c_tx = text_like_density(cgray)
        ch, cw = cgray.shape[:2]
        c_ar = cw / max(1, ch)
        metrics_center = {
            "ar_global": float(c_ar),
            "edge_density": float(c_ed),
            "text_density": float(c_tx),
        }
    else:
        metrics_center = {}

    return metrics_global, metrics_center


def doc_pass(metrics_global: dict,
             metrics_center: dict,
             card_like_ok: bool,
             require_quad: bool,
             ignore_ar: bool,
             edge_max_global: float,
             text_max_center: float) -> Tuple[bool, dict, dict]:
    # Global checks
    ar_ok_g = True if ignore_ar else (0.6 <= metrics_global["ar_global"] <= 2.0)
    edges_ok_g = (EDGE_MIN <= metrics_global["edge_density"] <= edge_max_global)
    text_ok_g = (TEXT_MIN <= metrics_global["text_density"] <= 0.30)

    global_checks = {
        "ar_ok": bool(ar_ok_g),
        "edges_ok": bool(edges_ok_g),
        "text_ok": bool(text_ok_g),
        "edge_cap_used": float(edge_max_global),
        "edge_min_used": float(EDGE_MIN),
        "text_cap_global": float(0.30 - 0.27),  # referencia histórica; no decisivo
    }

    # Center checks (si existen métricas)
    center_checks = {}
    if metrics_center:
        ar_ok_c = True if ignore_ar else (0.6 <= metrics_center["ar_global"] <= 2.0)
        edges_ok_c = (EDGE_MIN <= metrics_center["edge_density"] <= edge_max_global)
        text_ok_c = (TEXT_MIN <= metrics_center["text_density"] <= text_max_center)
        center_checks = {
            "ar_ok": bool(ar_ok_c),
            "edges_ok": bool(edges_ok_c),
            "text_ok": bool(text_ok_c),
            "text_cap_used": float(text_max_center),
            "text_min_used": float(TEXT_MIN),
        }

    # require_quad forzado
    if require_quad and not card_like_ok:
        return False, global_checks, center_checks

    # Aprobar si: tarjeta válida Y (global OK o center OK)
    either_ok = (
        (ar_ok_g and edges_ok_g and text_ok_g)
        or (center_checks != {} and center_checks["ar_ok"] and center_checks["edges_ok"] and center_checks["text_ok"])
    )

    return bool(card_like_ok and either_ok), global_checks, center_checks


def selfie_checks(rgb: np.ndarray) -> Tuple[bool, dict]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # brillo
    mean_brightness = float(gray.mean())
    if not (BRIGHT_MIN <= mean_brightness <= BRIGHT_MAX):
        return False, {"reason": "bad_brightness", "mean": mean_brightness}

    # rostro
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(60, 60))
    if len(faces) != 1:
        return False, {"reason": "face_count", "count": int(len(faces))}

    (x, y, w, h) = faces[0]
    area_ratio = (w * h) / float(rgb.shape[0] * rgb.shape[1])
    if area_ratio < FACE_MIN_AREA_RATIO:
        return False, {"reason": "face_too_small", "area_ratio": area_ratio}

    roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.08, minNeighbors=5, minSize=(18, 18))
    if len(eyes) < EYE_MIN_COUNT:
        # No se ven los ojos: gorra/gafas/ocluido -> inválido
        return False, {"reason": "eyes_not_visible", "eyes": int(len(eyes))}

    return True, {"reason": "ok", "face_area_ratio": area_ratio, "eyes": int(len(eyes))}


def deepface_verify(img1: np.ndarray, img2: np.ndarray,
                    threshold: float,
                    detector_pref: str) -> Tuple[bool, dict]:
    fallback_used = False
    note = ""
    model = "ArcFace"
    detector = detector_pref
    enforce = True

    def _run(detector_backend: str):
        return DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model,
            detector_backend=detector_backend,
            distance_metric="cosine",
            enforce_detection=True
        )

    try:
        res = _run(detector)
    except Exception as e:
        # fallback a opencv
        try:
            detector = "opencv"
            fallback_used = True
            note = f"{detector_pref} failed: {type(e).__name__}"
            res = _run(detector)
        except Exception as e2:
            return False, {
                "ok": False,
                "error": f"DeepFaceError: {type(e2).__name__}",
                "fallback_used": fallback_used,
                "note": note or str(e2),
                "model": model,
                "detector": detector,
                "enforce": enforce,
            }

    dist = float(res.get("distance", 1.0))
    verified = dist <= float(threshold)
    return verified, {
        "verified": bool(verified),
        "distance": dist,
        "threshold": float(threshold),
        "model": model,
        "detector": detector,
        "enforce": enforce,
        "fallback_used": fallback_used,
        "note": note,
    }


# =========================
# Endpoints
# =========================
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy", "version": VERSION}


@app.post("/debug")
async def debug(
    id_image: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    a = await id_image.read()
    b = await selfie.read()
    img_a = Image.open(io.BytesIO(a))
    img_b = Image.open(io.BytesIO(b))
    return {
        "ok": True,
        "id_image": {"bytes": len(a), "format": img_a.format, "size": img_a.size, "mode": img_a.mode},
        "selfie": {"bytes": len(b), "format": img_b.format, "size": img_b.size, "mode": img_b.mode},
        "version": VERSION
    }


@app.post("/why-doc")
async def why_doc(
    id_image: UploadFile = File(...),
    use_center_crop: bool = Form(False),
    require_quad: bool = Form(True),
    ignore_ar: bool = Form(False),
    edge_max_global: float = Form(EDGE_MAX_GLOBAL_DEF),
    text_max_center: float = Form(TEXT_MAX_CENTER_DEF),
):
    try:
        raw = await id_image.read()
        rgb = load_as_rgb_array(raw)

        # card-like
        card_ok, card_meta = doc_quadrilateral_check(rgb, ignore_ar, require_quad)

        # métricas
        mg, mc = doc_metrics(rgb, use_center_crop)

        # aprobación
        passed, gcheck, ccheck = doc_pass(
            mg, mc, card_ok, require_quad, ignore_ar, edge_max_global, text_max_center
        )

        return {
            "ok": True,
            "card_like": bool(card_ok),
            "card_meta": card_meta,
            "metrics_global": mg,
            "global_checks": gcheck,
            "version": VERSION,
            "relax": bool(ignore_ar),
            "use_center_crop": bool(use_center_crop),
            "metrics_center": mc,
            "center_checks": ccheck,
        }
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": f"{type(e).__name__}: {str(e)}", "version": VERSION})


@app.post("/why-selfie")
async def why_selfie(
    selfie: UploadFile = File(...)
):
    try:
        raw = await selfie.read()
        rgb = load_as_rgb_array(raw)
        ok, meta = selfie_checks(rgb)
        return {"ok": True, "selfie_ok": bool(ok), "selfie_meta": meta, "version": VERSION}
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": f"{type(e).__name__}: {str(e)}", "version": VERSION})


@app.post("/verify")
async def verify(
    id_image: UploadFile = File(..., description="Imagen del documento (frontal)"),
    selfie: UploadFile = File(..., description="Selfie a validar"),
    threshold: float = Form(DEFAULT_THRESHOLD),

    # Doc tunables
    use_center_crop: bool = Form(True),
    require_quad: bool = Form(True),
    ignore_ar: bool = Form(False),
    edge_max_global: float = Form(EDGE_MAX_GLOBAL_DEF),
    text_max_center: float = Form(TEXT_MAX_CENTER_DEF),
):
    try:
        # -------- load
        id_raw = await id_image.read()
        sf_raw = await selfie.read()
        id_rgb = load_as_rgb_array(id_raw)
        sf_rgb = load_as_rgb_array(sf_raw)

        # -------- doc checks
        card_ok, card_meta = doc_quadrilateral_check(id_rgb, ignore_ar, require_quad)
        mg, mc = doc_metrics(id_rgb, use_center_crop)
        doc_ok, gcheck, ccheck = doc_pass(
            mg, mc, card_ok, require_quad, ignore_ar, edge_max_global, text_max_center
        )

        # -------- selfie checks (estrictos)
        selfie_ok, selfie_meta = selfie_checks(sf_rgb)

        # Si falla cualquiera de las dos, no corremos biometría
        if not (doc_ok and selfie_ok):
            return {
                "ok": False,
                "doc_ok": bool(doc_ok),
                "selfie_ok": bool(selfie_ok),
                "reason": {
                    "doc": "ok" if doc_ok else "not_card_like",
                    "selfie": selfie_meta.get("reason", "invalid") if not selfie_ok else "ok",
                },
                "selfie_card_like": False,
                "card_meta": card_meta,
                "loose_meta": {"loose": ignore_ar, "criteria_global": gcheck if gcheck else "",
                               "used_center_crop": use_center_crop,
                               "criteria_center": ccheck if ccheck else ""},
                "strict_meta": {"strict_card_like": bool(card_ok and not require_quad is False),
                                "strict_doc_ok_face": bool(selfie_ok)},
                "doc_face_reason": selfie_meta.get("reason", "") if not selfie_ok else "",
                "doc_mode_used": "auto_failed",
                "relax": bool(ignore_ar),
                "use_center_crop": bool(use_center_crop),
                "version": VERSION,
            }

        # -------- biometrics
        verified, bio = deepface_verify(id_rgb, sf_rgb, threshold, APP_DETECTOR)

        return {
            "ok": bool(verified),
            **bio,
            "doc_ok": True,
            "selfie_ok": True,
            "card_meta": {**card_meta, "loose": ignore_ar, "criteria_global": gcheck if gcheck else "",
                          "used_center_crop": use_center_crop, "criteria_center": ccheck if ccheck else ""},
            "loose_meta": {"loose": ignore_ar, "criteria_global": gcheck if gcheck else "",
                           "used_center_crop": use_center_crop, "criteria_center": ccheck if ccheck else ""},
            "strict_meta": None if ignore_ar else {"strict_card_like": bool(card_ok), "strict_doc_ok_face": True},
            "doc_mode_used": "loose" if ignore_ar else ("strict" if require_quad else "hybrid"),
            "relax": bool(ignore_ar),
            "use_center_crop": bool(use_center_crop),
            "version": VERSION,
            "text_cap_used": float(text_max_center),
            "text_min_used": float(TEXT_MIN),
            "edge_cap_used": float(edge_max_global),
            "edge_min_used": float(EDGE_MIN),
            "require_quad": bool(require_quad),
            "selfie_not_card": True,  # explicitamos que no exigimos forma de tarjeta en selfie
        }

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": f"{type(e).__name__}: {str(e)}", "version": VERSION}
        )


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
