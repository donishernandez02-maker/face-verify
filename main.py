# main.py  (v1.6.3)
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
APP_VERSION           = "1.6.3"
APP_DETECTOR_DEFAULT  = "auto"       # auto | retinaface | opencv | mtcnn | ssd | yolov8 | mediapipe ...
DEFAULT_THRESHOLD     = float(os.getenv("ARC_THRESHOLD", "0.38"))
DOC_MODE_DEFAULT      = "auto"       # auto (strict->loose) | strict | loose

# Heurísticas de documento (ID card) - base
CARD_MIN_AREA_FRAC_BASE = 0.06
CARD_MAX_AREA_FRAC_BASE = 0.985
CARD_AR_MIN_BASE        = 0.80
CARD_AR_MAX_BASE        = 2.60
CARD_ANGLE_TOL_BASE     = 40.0

# Selfie / Face
MIN_FACE_FRAC         = 0.12   # cara ≥12% del lado menor del frame
MAX_SELFIE_FACES      = 1      # exactamente una cara en selfie

# Loose global
AR_LOOSE_MIN_G        = 0.75
AR_LOOSE_MAX_G        = 2.80
EDGE_DENS_MIN_G       = 0.006
EDGE_DENS_MAX_G       = 0.33
TEXT_DENS_MIN_G       = 0.0012
TEXT_DENS_MAX_G       = 0.030

# Loose center (más permisivo si el centro está “limpio”)
AR_LOOSE_MIN_C        = 0.70
AR_LOOSE_MAX_C        = 3.00
EDGE_DENS_MAX_C       = 0.50
TEXT_DENS_MAX_C       = 0.08
CENTER_FRACTION       = 0.70   # 70% central

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

def _quad_is_rect_like(cnt: np.ndarray, angle_tol: float) -> bool:
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
        if abs(90 - ang) > angle_tol:
            return False
    return True

def _prep_edges(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(blur)
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(eq, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(closed, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    return edges

def detect_card_quad(bgr: np.ndarray, area_min_frac: float, area_max_frac: float,
                     ar_min: float, ar_max: float, angle_tol: float) -> Tuple[bool, Dict]:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = _prep_edges(gray)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, {"reason": "no_contours"}
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    area_img = w * h
    for c in cnts[:25]:
        area = cv2.contourArea(c)
        if area <= 1:
            continue
        frac = area / (area_img + 1e-9)
        if frac < area_min_frac or frac > area_max_frac:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)
        if len(approx) == 4:
            x, y, ww, hh = cv2.boundingRect(approx)
            ar = ww / float(hh + 1e-9)
            rect_like = _quad_is_rect_like(approx, angle_tol)
            if (ar_min <= ar <= ar_max) and rect_like:
                return True, {"area_frac": frac, "ar": ar, "rect_like": True}
    return False, {"reason": "no_quad_like_card"}

def edge_density(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    e = _prep_edges(gray)
    h, w = e.shape[:2]
    return float(np.count_nonzero(e)) / float(h * w + 1e-9)

def text_density_mser(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=5000)
    except TypeError:
        mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros_like(gray, dtype=np.uint8)
    h, w = gray.shape[:2]
    area_img = float(h * w)
    for r in regions[:3000]:
        x, y, ww, hh = cv2.boundingRect(r.reshape(-1, 1, 2))
        if ww * hh < 60 or ww * hh > 20000:
            continue
        ar = ww / float(hh + 1e-9)
        if ar < 0.5 or ar > 12.0:
            continue
        cv2.rectangle(mask, (x, y), (x + ww, y + hh), 255, -1)
    dens = float(np.count_nonzero(mask)) / (area_img + 1e-9)
    return dens

def crop_center(img_rgb: np.ndarray, frac: float = CENTER_FRACTION) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    cw, ch = int(w * frac), int(h * frac)
    x1 = (w - cw) // 2
    y1 = (h - ch) // 2
    return img_rgb[y1:y1+ch, x1:x1+cw]

def compute_metrics(img_rgb: np.ndarray) -> Dict:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]
    return {
        "ar_global": w / float(h + 1e-9),
        "edge_density": edge_density(bgr),
        "text_density": text_density_mser(bgr)
    }

def detect_card_any_rotation(img_rgb: np.ndarray, area_min_frac: float, area_max_frac: float,
                             ar_min: float, ar_max: float, angle_tol: float):
    candidates = []
    for k in (0, 1, 2, 3):
        test_rgb = np.rot90(img_rgb, k=k) if k else img_rgb
        bgr = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2BGR)
        ok, meta = detect_card_quad(bgr, area_min_frac, area_max_frac, ar_min, ar_max, angle_tol)
        meta = meta or {}
        meta['rotation'] = k * 90
        if ok:
            return True, meta
        candidates.append((meta.get('area_frac', 0.0), meta))
    best = max(candidates, key=lambda x: x[0])[1] if candidates else {"rotation": 0}
    return False, best

def extract_faces(img_rgb: np.ndarray, backend: str) -> List[dict]:
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

@app.post("/why-doc")
async def why_doc(
    id_image: UploadFile = File(..., description="Imagen del documento (frontal)"),
    relax: bool = Form(False),
    use_center_crop: bool = Form(False)
):
    try:
        # Ajustes según relax
        CARD_MIN_AREA_FRAC = CARD_MIN_AREA_FRAC_BASE * (0.8 if relax else 1.0)
        CARD_MAX_AREA_FRAC = CARD_MAX_AREA_FRAC_BASE
        CARD_AR_MIN        = CARD_AR_MIN_BASE  - (0.05 if relax else 0.0)
        CARD_AR_MAX        = CARD_AR_MAX_BASE  + (0.20 if relax else 0.0)
        CARD_ANGLE_TOL     = CARD_ANGLE_TOL_BASE + (5.0 if relax else 0.0)

        id_bytes = await id_image.read()
        id_pil = auto_orient(Image.open(io.BytesIO(id_bytes)))
        id_pil = upscale_if_tiny(id_pil, 500)
        id_arr_rgb = pil_to_array(id_pil)

        # tarjeta rotation-aware
        card_like, card_meta = detect_card_any_rotation(
            id_arr_rgb,
            CARD_MIN_AREA_FRAC, CARD_MAX_AREA_FRAC,
            CARD_AR_MIN, CARD_AR_MAX, CARD_ANGLE_TOL
        )

        # métricas
        m_global = compute_metrics(id_arr_rgb)
        m_center = {}
        if use_center_crop:
            m_center = compute_metrics(crop_center(id_arr_rgb, CENTER_FRACTION))

        out = {
            "ok": True,
            "card_like": bool(card_like),
            "card_meta": card_meta,
            "metrics_global": m_global,
            "version": APP_VERSION,
            "relax": bool(relax),
            "use_center_crop": bool(use_center_crop)
        }
        if use_center_crop:
            out["metrics_center"] = m_center
        return out
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)}", "version": APP_VERSION}

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
    detector:  str   = Form(APP_DETECTOR_DEFAULT),
    enforce:   bool  = Form(True),
    doc_mode:  str   = Form(DOC_MODE_DEFAULT),
    relax:     bool  = Form(False),
    use_center_crop: bool = Form(False)
) -> Dict:
    try:
        # Ajustes según relax
        CARD_MIN_AREA_FRAC = CARD_MIN_AREA_FRAC_BASE * (0.8 if relax else 1.0)
        CARD_MAX_AREA_FRAC = CARD_MAX_AREA_FRAC_BASE
        CARD_AR_MIN        = CARD_AR_MIN_BASE  - (0.05 if relax else 0.0)
        CARD_AR_MAX        = CARD_AR_MAX_BASE  + (0.20 if relax else 0.0)
        CARD_ANGLE_TOL     = CARD_ANGLE_TOL_BASE + (5.0 if relax else 0.0)

        # 1) Cargar
        id_bytes = await id_image.read()
        sf_bytes = await selfie.read()
        id_pil = auto_orient(Image.open(io.BytesIO(id_bytes)))
        sf_pil = auto_orient(Image.open(io.BytesIO(sf_bytes)))
        id_pil = upscale_if_tiny(id_pil, 500)
        sf_pil = upscale_if_tiny(sf_pil, 500)
        id_arr_rgb = pil_to_array(id_pil)
        sf_arr_rgb = pil_to_array(sf_pil)

        # 2) Documento
        card_like, card_meta = detect_card_any_rotation(
            id_arr_rgb,
            CARD_MIN_AREA_FRAC, CARD_MAX_AREA_FRAC,
            CARD_AR_MIN, CARD_AR_MAX, CARD_ANGLE_TOL
        )

        # caras en doc
        doc_faces = []
        try:
            doc_faces = extract_faces(id_arr_rgb, backend="retinaface")
        except Exception:
            try:
                doc_faces = extract_faces(id_arr_rgb, backend="opencv")
            except Exception:
                doc_faces = []

        doc_h, doc_w = id_arr_rgb.shape[:2]
        doc_ok_face = False
        if doc_faces:
            f = pick_best_face(doc_faces, prefer_larger=False)
            if f is not None:
                try:
                    fa = sorted(doc_faces, key=lambda d: d.get("confidence", 0), reverse=True)[0].get("facial_area", {})
                    fw, fh = fa.get("w", 0), fa.get("h", 0)
                    min_side = min(doc_w, doc_h)
                    doc_ok_face = max(fw, fh) <= 0.45 * float(min_side)
                except Exception:
                    doc_ok_face = True

        # decisiones
        doc_mode_req = (doc_mode or DOC_MODE_DEFAULT).lower().strip()

        def strict_decision() -> bool:
            return bool(card_like and doc_ok_face)

        def loose_decision() -> Tuple[bool, Dict]:
            bgr_full = cv2.cvtColor(id_arr_rgb, cv2.COLOR_RGB2BGR)
            ar_g = doc_w / float(doc_h + 1e-9)
            d_edge_g = edge_density(bgr_full)
            d_text_g = text_density_mser(bgr_full)

            ar_ok_g    = AR_LOOSE_MIN_G <= ar_g <= AR_LOOSE_MAX_G
            edges_ok_g = EDGE_DENS_MIN_G <= d_edge_g <= EDGE_DENS_MAX_G
            text_ok_g  = TEXT_DENS_MIN_G <= d_text_g  <= TEXT_DENS_MAX_G

            ok_global = bool( (doc_ok_face and ar_ok_g and edges_ok_g) or text_ok_g )

            ok_center = False
            criteria_center = None
            if use_center_crop:
                center_rgb = crop_center(id_arr_rgb, CENTER_FRACTION)
                ch, cw = center_rgb.shape[:2]
                ar_c = cw / float(ch + 1e-9)
                bgr_c = cv2.cvtColor(center_rgb, cv2.COLOR_RGB2BGR)
                d_edge_c = edge_density(bgr_c)
                d_text_c = text_density_mser(bgr_c)

                ar_ok_c    = AR_LOOSE_MIN_C <= ar_c <= AR_LOOSE_MAX_C
                edges_ok_c = d_edge_c <= EDGE_DENS_MAX_C
                text_ok_c  = d_text_c <= TEXT_DENS_MAX_C

                criteria_center = {
                    "ar": ar_c, "edge_density": d_edge_c, "text_density": d_text_c,
                    "ar_ok": ar_ok_c, "edges_ok": edges_ok_c, "text_ok": text_ok_c
                }

                ok_center = bool( (doc_ok_face and ar_ok_c and edges_ok_c) or text_ok_c )

            ok = bool(ok_global or ok_center)
            meta_patch = {
                "loose": True,
                "criteria_global": {
                    "ar": ar_g, "edge_density": d_edge_g, "text_density": d_text_g,
                    "doc_ok_face": doc_ok_face,
                    "ar_ok": ar_ok_g, "edges_ok": edges_ok_g, "text_ok": text_ok_g
                },
                "used_center_crop": bool(use_center_crop),
            }
            if criteria_center is not None:
                meta_patch["criteria_center"] = criteria_center
            return ok, meta_patch

        doc_mode_used = None
        loose_meta = None
        if doc_mode_req == "strict":
            doc_ok = strict_decision()
            doc_mode_used = "strict"
        elif doc_mode_req == "loose":
            doc_ok, loose_meta = loose_decision()
            doc_mode_used = "loose"
            if doc_ok:
                card_meta = {**(card_meta or {}), **(loose_meta or {})}
        else:
            doc_ok = strict_decision()
            if doc_ok:
                doc_mode_used = "strict"
            else:
                doc_ok, loose_meta = loose_decision()
                if doc_ok:
                    doc_mode_used = "loose"
                    card_meta = {**(card_meta or {}), **(loose_meta or {})}
                else:
                    doc_mode_used = "auto_failed"

        # 3) Selfie
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
                "loose_meta": (loose_meta or {}),
                "doc_mode_used": doc_mode_used,
                "relax": bool(relax),
                "use_center_crop": bool(use_center_crop),
                "version": APP_VERSION
            }

        # 4) Verificación (cara de doc vs selfie)
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

        ver.update({
            "doc_ok": True,
            "selfie_ok": True,
            "card_meta": card_meta,
            "loose_meta": (loose_meta or {}),
            "doc_mode_used": doc_mode_used,
            "relax": bool(relax),
            "use_center_crop": bool(use_center_crop),
            "version": APP_VERSION
        })
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
