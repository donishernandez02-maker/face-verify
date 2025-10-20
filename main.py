# main.py — Face Verify API (v1.8.0)
import io
import os
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

# OpenCV es base de los heurísticos (bordes / contornos / mser-like)
import cv2

# =========================
# Config & Constantes
# =========================
APP_VERSION = "1.8.0"

# Detector por defecto para DeepFace
APP_DETECTOR_DEFAULT = os.getenv("DETECTOR_DEFAULT", "opencv")

# Umbral biométrico por defecto (cosine distance para ArcFace)
DEFAULT_THRESHOLD = float(os.getenv("ARC_THRESHOLD", "0.38"))

# Límites heurísticos de documento (puedes tunear por env)
# Densidad de bordes global (cap superior) y mínimo
EDGE_DENS_MAX_G = float(os.getenv("EDGE_DENS_MAX_G", "0.55"))
EDGE_DENS_MIN_G = float(os.getenv("EDGE_DENS_MIN_G", "0.01"))

# Densidad de “texto” global (cap superior) — muy bajo en promedio del frame completo
TEXT_DENS_MAX_G = float(os.getenv("TEXT_DENS_MAX_G", "0.03"))

# Densidad de “texto” en center-crop (cap superior) y mínimo (para no ser totalmente plano)
TEXT_DENS_MAX_C = float(os.getenv("TEXT_DENS_MAX_C", "0.18"))
TEXT_DENS_MIN_C = float(os.getenv("TEXT_DENS_MIN_C", "0.01"))

# Aspect ratio esperado (tarjetas habituales ~54mm x 86mm => ~0.63 si usas H/W).
# Lo usamos solo si ignore_ar=False
AR_MIN = float(os.getenv("AR_MIN", "0.5"))
AR_MAX = float(os.getenv("AR_MAX", "0.8"))

# Center-crop factor (porción central a evaluar)
CENTER_CROP_F = float(os.getenv("CENTER_CROP_F", "0.6"))

# Selfie: tamaño mínimo de rostro relativo a la altura de la imagen
MIN_FACE_RATIO_DEFAULT = float(os.getenv("MIN_FACE_RATIO_DEFAULT", "0.33"))

# =========================
# FastAPI
# =========================
app = FastAPI(title="Face Verify API", version=APP_VERSION)

# CORS libre para pruebas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Utilidades de imagen
# =========================
def exif_transpose(pil_img: Image.Image) -> Image.Image:
    """Corrige orientación según EXIF (parche vertical/horizontal)."""
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img

def pil_to_array(img: Image.Image) -> np.ndarray:
    img = exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def center_crop(img: np.ndarray, frac: float = CENTER_CROP_F) -> np.ndarray:
    h, w = img.shape[:2]
    ch = int(h * frac)
    cw = int(w * frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0:y0+ch, x0:x0+cw]

def rotate_image(img: np.ndarray, rot: int) -> np.ndarray:
    if rot % 360 == 0:
        return img
    if rot % 360 == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot % 360 == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot % 360 == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def rotate_image_bound(img: np.ndarray, angle: float) -> np.ndarray:
    """Rota manteniendo contenido completo (no recorta)."""
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[1, 0])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# =========================
# Métricas heurísticas (doc)
# =========================
def edge_density(gray: np.ndarray) -> float:
    """Proporción de pixeles detectados como borde (Canny)."""
    v = max(50, int(np.median(gray) * 0.66))
    edges = cv2.Canny(gray, v, v * 2)
    return float((edges > 0).mean())

def text_like_density(gray: np.ndarray) -> float:
    """
    Medida simple de “texto”: binarizamos y contamos contornos pequeños/medianos.
    Normalizamos por el área; no es OCR, solo textura fina tipo caracteres.
    """
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 5
    )
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]
    h_area = h * w
    useful = [a for a in areas if (h_area * 0.00002) < a < (h_area * 0.005)]
    dens = len(useful) / (h * w / 10000.0)  # normaliza
    dens = min(1.0, dens / 50.0)  # acotar aprox [0,1]
    return float(dens)

def aspect_ratio_h_over_w(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    return float(h) / float(max(1, w))

def find_quad_like_card(img: np.ndarray) -> Tuple[bool, dict]:
    """
    Busca el contorno mayor que se aproxime a un cuadrilátero razonable.
    Retorna (ok, meta).
    """
    gray = to_gray(img)
    v = max(50, int(np.median(gray) * 0.66))
    edges = cv2.Canny(gray, v, v * 2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, {"reason": "no_contours"}

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    h, w = gray.shape[:2]
    frame_area = h * w
    for c in cnts:
        area = cv2.contourArea(c)
        if area < frame_area * 0.05:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, cw, ch = cv2.boundingRect(approx)
            ar_hw = ch / float(cw) if cw > 0 else 0.0
            if 0.5 <= ar_hw <= 0.8 or 0.5 <= (1.0 / max(1e-6, ar_hw)) <= 0.8:
                return True, {"reason": "ok", "bbox": [int(x), int(y), int(cw), int(ch)]}
    return False, {"reason": "no_quad_like_card"}

def compute_doc_metrics(img: np.ndarray) -> Tuple[dict, dict]:
    """Calcula métricas globales y del center-crop."""
    gray = to_gray(img)
    ar_g = aspect_ratio_h_over_w(img)
    e_g = edge_density(gray)
    t_g = text_like_density(gray)

    crop = center_crop(img, CENTER_CROP_F)
    gray_c = to_gray(crop)
    ar_c = aspect_ratio_h_over_w(crop)
    e_c = edge_density(gray_c)
    t_c = text_like_density(gray_c)

    metrics_global = {
        "ar_global": float(ar_g),
        "edge_density": float(e_g),
        "text_density": float(t_g),
    }
    metrics_center = {
        "ar_global": float(ar_c),
        "edge_density": float(e_c),
        "text_density": float(t_c),
    }
    return metrics_global, metrics_center

def evaluate_doc_checks(metrics_global: dict,
                        metrics_center: dict,
                        require_quad: bool,
                        ignore_ar: bool,
                        text_cap_center: float,
                        edge_cap_global: float,
                        quad_like: bool) -> Tuple[bool, dict, dict]:
    """
    Retorna (doc_ok, checks_global, checks_center)
    """
    ar_g = metrics_global["ar_global"]
    e_g = metrics_global["edge_density"]
    t_g = metrics_global["text_density"]

    ar_c = metrics_center["ar_global"]
    e_c = metrics_center["edge_density"]
    t_c = metrics_center["text_density"]

    edges_ok_g = (EDGE_DENS_MIN_G <= e_g <= edge_cap_global)
    text_ok_g = (t_g <= TEXT_DENS_MAX_G)

    text_ok_c = (TEXT_DENS_MIN_C <= t_c <= text_cap_center)
    edges_ok_c = True  # no rígido en center

    ar_ok_g = True if ignore_ar else (AR_MIN <= ar_g <= AR_MAX)
    ar_ok_c = True if ignore_ar else (AR_MIN <= ar_c <= AR_MAX)

    quad_ok = (True if not require_quad else quad_like)

    doc_ok = (quad_ok and edges_ok_g and text_ok_g and text_ok_c and ar_ok_g and ar_ok_c)

    checks_global = {
        "ar_ok": bool(ar_ok_g),
        "edges_ok": bool(edges_ok_g),
        "text_ok": bool(text_ok_g),
        "edge_cap_used": float(edge_cap_global),
        "edge_min_used": float(EDGE_DENS_MIN_G),
        "text_cap_global": float(TEXT_DENS_MAX_G),
    }
    checks_center = {
        "ar_ok": bool(ar_ok_c),
        "edges_ok": bool(edges_ok_c),
        "text_ok": bool(text_ok_c),
        "text_cap_used": float(text_cap_center),
        "text_min_used": float(TEXT_DENS_MIN_C),
    }
    return doc_ok, checks_global, checks_center

def best_rotation_card(img: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Prueba rotaciones 0/90/180/270 y devuelve la primera que cumpla card-like.
    Si ninguna, retorna rot=0 y razón encontrada.
    """
    tried = []
    for rot in [0, 90, 180, 270]:
        rr = rotate_image(img, rot)
        ok, meta = find_quad_like_card(rr)
        if ok:
            meta_out = {"reason": "ok", "rotation": float(rot)}
            meta_out.update(meta)
            return rr, meta_out
        tried.append(meta.get("reason", "no_card"))
    return img, {"reason": tried[-1] if tried else "no_quad_like_card", "rotation": 0}

def deepface_single_face(img: np.ndarray, detector_backend: str = "opencv") -> Tuple[bool, dict]:
    """
    Chequea si hay exactamente una cara en el documento (para evitar “cualquier foto”).
    """
    try:
        dets = DeepFace.extract_faces(img_path=img, detector_backend=detector_backend,
                                      enforce_detection=False, align=False) or []
    except Exception:
        dets = []
    if len(dets) != 1:
        return False, {"reason": "face_count", "count": len(dets)}
    fa = dets[0].get("facial_area") or dets[0].get("region") or {}
    h, w = img.shape[:2]
    area_ratio = float(fa.get("h", 0)) / float(h if h else 1)
    return True, {"reason": "ok", "area_ratio": area_ratio}

def check_document_like(img: np.ndarray,
                        use_center_crop: bool,
                        ignore_ar: bool,
                        doc_mode: str,
                        require_quad: bool,
                        text_max_center: float,
                        edge_max_global: float) -> Tuple[bool, dict, dict, dict, str]:
    """
    Valida doc con dos niveles:
      - loose: métricas (edges/text/AR) + (opcional) quad
      - strict: SOLO requiere que sea card-like (quad) y que haya exactamente 1 rostro en el doc.
    Retorna:
      - doc_ok (bool)
      - card_meta (dict)
      - loose_meta (dict)
      - strict_meta (dict)
      - doc_face_reason (str)
    """
    rotated, card_meta = best_rotation_card(img)

    mg, mc = compute_doc_metrics(rotated)
    used_center = False
    if use_center_crop:
        _ = center_crop(rotated, CENTER_CROP_F)
        used_center = True

    quad_like = (card_meta.get("reason") == "ok")

    loose_ok, checks_g, checks_c = evaluate_doc_checks(
        mg, mc,
        require_quad=require_quad,
        ignore_ar=ignore_ar,
        text_cap_center=text_max_center,
        edge_cap_global=edge_max_global,
        quad_like=quad_like
    )

    # Para strict, exigimos 1 rostro en doc (usando detector por defecto)
    doc_face_ok, face_meta = deepface_single_face(rotated, detector_backend=APP_DETECTOR_DEFAULT)

    strict_decision = (quad_like and doc_face_ok)
    strict_meta = {
        "strict_card_like": bool(quad_like),
        "strict_doc_ok_face": bool(doc_face_ok),
    }
    loose_meta = {
        "loose": bool(loose_ok),
        "criteria_global": checks_g,
        "used_center_crop": bool(used_center),
        "criteria_center": checks_c,
    }
    doc_face_reason = "ok" if doc_face_ok else face_meta.get("reason", "face_count")

    if doc_mode == "strict":
        return strict_decision, card_meta, loose_meta, strict_meta, ("" if strict_decision else doc_face_reason)
    if doc_mode == "loose":
        return loose_ok, card_meta, loose_meta, strict_meta, ("" if loose_ok else doc_face_reason)
    # auto = strict
    return strict_decision, card_meta, loose_meta, strict_meta, ("" if strict_decision else doc_face_reason)

# =========================
# Face crops helpers (nuevo)
# =========================
def extract_single_face_np(img_rgb: np.ndarray, detector_backend: str):
    """
    Devuelve (face_rgb_uint8, meta) o (None, {'reason': ...})
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True
        )
    except Exception as e:
        return None, {"reason": type(e).__name__}

    if not faces:
        return None, {"reason": "face_count", "count": 0}

    # elegir la cara más grande
    best = max(
        faces,
        key=lambda f: (f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0))
    )
    face = best.get("face")
    fa = best.get("facial_area", {})
    if face is None:
        return None, {"reason": "no_face_crop"}

    # DeepFace devuelve face en RGB [0..1]
    if face.dtype != np.uint8:
        face = (np.clip(face, 0, 1) * 255).astype("uint8")

    meta = {"reason": "ok", "bbox": [fa.get("x"), fa.get("y"), fa.get("w"), fa.get("h")]}
    return face, meta

# =========================
# Selfie helpers
# =========================
def detect_face_and_ratio(img: np.ndarray, detector_backend: str = "opencv",
                          tight_crop: bool = False) -> Tuple[bool, dict, Optional[np.ndarray]]:
    """
    Devuelve (ok, meta_dict, face_img or None)
    meta_dict: {"reason": "ok"|"face_count"|"face_too_small", "area_ratio": float, "count": int}
    area_ratio = alto_cara / alto_imagen
    """
    try:
        detections = DeepFace.extract_faces(
            img_path=img,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=False
        ) or []
    except Exception:
        detections = []

    if len(detections) != 1:
        return False, {"reason": "face_count", "count": len(detections), "area_ratio": 0.0}, None

    det = detections[0]
    fa = det.get("facial_area") or det.get("region") or {}
    x, y = int(fa.get("x", 0)), int(fa.get("y", 0))
    w, h = int(fa.get("w", 0)), int(fa.get("h", 0))
    H, W = img.shape[:2]
    area_ratio = h / float(H) if H > 0 else 0.0

    face_img = None
    if tight_crop and w > 0 and h > 0:
        x2 = max(0, x); y2 = max(0, y)
        x3 = min(W, x + w); y3 = min(H, y + h)
        if (y2 < y3) and (x2 < x3):
            face_img = img[y2:y3, x2:x3, :].copy()

    return True, {"reason": "ok", "area_ratio": float(area_ratio)}, face_img

# =========================
# ENDPOINTS
# =========================
@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy", "version": APP_VERSION}

@app.post("/debug")
async def debug(
    id_image: UploadFile = File(None),
    selfie: UploadFile = File(None),
):
    out = {"ok": True, "version": APP_VERSION}
    if id_image is not None:
        b = await id_image.read()
        try:
            im = Image.open(io.BytesIO(b))
            out["id_image"] = {
                "bytes": len(b),
                "format": im.format,
                "size": list(im.size),
                "mode": im.mode,
            }
        except Exception as e:
            out["id_image"] = {"error": str(e)}
    if selfie is not None:
        b2 = await selfie.read()
        try:
            im2 = Image.open(io.BytesIO(b2))
            out["selfie"] = {
                "bytes": len(b2),
                "format": im2.format,
                "size": list(im2.size),
                "mode": im2.mode,
            }
        except Exception as e:
            out["selfie"] = {"error": str(e)}
    return out

@app.post("/why-doc")
async def why_doc(
    id_image: UploadFile = File(..., description="Imagen del documento (frontal)"),
    use_center_crop: bool = Form(False),
    require_quad: bool = Form(True),
    ignore_ar: bool = Form(False),
    text_max_center: float = Form(TEXT_DENS_MAX_C),
    edge_max_global: float = Form(EDGE_DENS_MAX_G),
):
    try:
        b = await id_image.read()
        pil = Image.open(io.BytesIO(b))
        arr = pil_to_array(pil)

        rotated, card_meta = best_rotation_card(arr)
        mg, mc = compute_doc_metrics(rotated)

        doc_ok, checks_g, checks_c = evaluate_doc_checks(
            mg, mc,
            require_quad=require_quad,
            ignore_ar=ignore_ar,
            text_cap_center=text_max_center,
            edge_cap_global=edge_max_global,
            quad_like=(card_meta.get("reason") == "ok")
        )

        out = {
            "ok": True,
            "card_like": bool(card_meta.get("reason") == "ok"),
            "card_meta": card_meta,
            "metrics_global": mg,
            "global_checks": checks_g,
            "version": APP_VERSION,
            "relax": bool(ignore_ar),
            "use_center_crop": bool(use_center_crop),
        }

        if use_center_crop:
            cc = center_crop(rotated, CENTER_CROP_F)
            mg_c, mc_c = compute_doc_metrics(cc)
            _, _, checks_center = evaluate_doc_checks(
                mg_c, mc_c,
                require_quad=False,
                ignore_ar=ignore_ar,
                text_cap_center=text_max_center,
                edge_cap_global=edge_max_global,
                quad_like=True
            )
            out["metrics_center"] = mg_c
            out["center_checks"] = checks_center

        return out
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": f"{type(e).__name__}: {str(e)}", "version": APP_VERSION})

@app.post("/why-selfie")
async def why_selfie(
    selfie: UploadFile = File(..., description="Selfie a validar"),
    detector: str = Form(APP_DETECTOR_DEFAULT),
):
    try:
        b = await selfie.read()
        pil = Image.open(io.BytesIO(b))
        arr = pil_to_array(pil)
        ok, meta, _ = detect_face_and_ratio(arr, detector_backend=detector, tight_crop=False)
        return {"ok": True, "selfie_ok": bool(ok and meta.get("reason") == "ok"), "selfie_meta": meta, "version": APP_VERSION}
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": f"{type(e).__name__}: {str(e)}", "version": APP_VERSION})

@app.post("/verify")
async def verify(
    id_image: UploadFile = File(..., description="Imagen del documento (frontal)"),
    selfie: UploadFile = File(..., description="Selfie a validar"),
    threshold: float = Form(DEFAULT_THRESHOLD),

    # Documento
    use_center_crop: bool = Form(False),
    ignore_ar: bool = Form(False),
    doc_mode: str = Form("auto"),       # strict | loose | auto (auto usa strict)
    require_quad: bool = Form(True),
    text_max_center: float = Form(TEXT_DENS_MAX_C),
    edge_max_global: float = Form(EDGE_DENS_MAX_G),

    # Detector + DeepFace
    detector: str = Form(APP_DETECTOR_DEFAULT),
    enforce: bool = Form(False),

    # Selfie params
    min_face_ratio_selfie: float = Form(MIN_FACE_RATIO_DEFAULT),
    auto_tight_crop: bool = Form(False),
) -> Dict:
    """
    Flujo completo:
    1) Validar documento (card-like + métricas + 1 rostro).
    2) Validar selfie (1 rostro y tamaño mínimo relativo al alto).
    3) Si ambos ok, comparar biometría (ArcFace) sobre CROPS de rostro.
    """
    try:
        # Leer imágenes
        id_bytes = await id_image.read()
        sf_bytes = await selfie.read()

        id_pil = Image.open(io.BytesIO(id_bytes))
        sf_pil = Image.open(io.BytesIO(sf_bytes))

        id_arr = pil_to_array(id_pil)
        sf_arr = pil_to_array(sf_pil)

        # 1) Documento
        doc_ok, card_meta, loose_meta, strict_meta, doc_face_reason = check_document_like(
            id_arr,
            use_center_crop=use_center_crop,
            ignore_ar=ignore_ar,
            doc_mode=doc_mode,
            require_quad=require_quad,
            text_max_center=text_max_center,
            edge_max_global=edge_max_global,
        )

        # 2) Selfie (detección básica + ratio y opcional crop ajustado)
        s_ok, s_meta, face_img = detect_face_and_ratio(
            sf_arr,
            detector_backend=detector,
            tight_crop=auto_tight_crop
        )
        selfie_card_like = False  # no se valida card-like en selfie

        if not s_ok:
            return {
                "ok": False,
                "doc_ok": bool(doc_ok),
                "selfie_ok": False,
                "reason": {"doc": "ok" if doc_ok else (card_meta.get("reason", "doc_fail")), "selfie": s_meta.get("reason", "face_count")},
                "selfie_card_like": selfie_card_like,
                "card_meta": card_meta,
                "loose_meta": loose_meta,
                "strict_meta": strict_meta,
                "doc_face_reason": s_meta.get("reason"),
                "doc_mode_used": "auto_failed" if (doc_mode == "auto" and not doc_ok) else doc_mode,
                "relax": bool(ignore_ar),
                "use_center_crop": bool(use_center_crop),
                "version": APP_VERSION,
                "selfie_meta": s_meta,
            }

        area_ratio = float(s_meta.get("area_ratio", 0.0))
        if area_ratio < float(min_face_ratio_selfie):
            return {
                "ok": False,
                "doc_ok": bool(doc_ok),
                "selfie_ok": False,
                "reason": {"doc": "ok" if doc_ok else (card_meta.get("reason", "doc_fail")), "selfie": "face_too_small"},
                "selfie_card_like": selfie_card_like,
                "card_meta": card_meta,
                "loose_meta": loose_meta,
                "strict_meta": strict_meta,
                "doc_face_reason": "face_too_small",
                "doc_mode_used": "auto_failed" if (doc_mode == "auto" and not doc_ok) else doc_mode,
                "relax": bool(ignore_ar),
                "use_center_crop": bool(use_center_crop),
                "version": APP_VERSION,
                "selfie_meta": {"reason": "face_too_small", "area_ratio": area_ratio},
            }

        # Si documento NO pasó, cortamos aquí
        if not doc_ok:
            return {
                "ok": False,
                "doc_ok": False,
                "selfie_ok": True,
                "reason": {"doc": card_meta.get("reason", "doc_fail"), "selfie": "ok"},
                "selfie_card_like": selfie_card_like,
                "card_meta": card_meta,
                "loose_meta": loose_meta,
                "strict_meta": strict_meta,
                "doc_face_reason": doc_face_reason,
                "doc_mode_used": "auto_failed" if doc_mode == "auto" else doc_mode,
                "relax": bool(ignore_ar),
                "use_center_crop": bool(use_center_crop),
                "version": APP_VERSION,
                "selfie_meta": s_meta,
            }

        # =========================
        # 3) Verificación biométrica — comparar CARA vs CARA
        # =========================

        # 3.a rotar doc si card_meta trae rotación (usar rotate bound para conservar contenido)
        rotation = float(card_meta.get("rotation", 0.0)) if isinstance(card_meta, dict) else 0.0
        doc_for_face = id_arr
        if abs(rotation) > 1e-3:
            # convertir a BGR -> rotar -> volver a RGB
            doc_bgr = cv2.cvtColor(id_arr, cv2.COLOR_RGB2BGR)
            doc_bgr = rotate_image_bound(doc_bgr, rotation)
            doc_for_face = cv2.cvtColor(doc_bgr, cv2.COLOR_BGR2RGB)

        # 3.b extraer UNA cara del doc y de la selfie (con el MISMO detector)
        doc_face, doc_meta_face = extract_single_face_np(doc_for_face, detector_backend=detector)
        if face_img is not None:
            selfie_face = face_img
            selfie_meta_face = {"reason": "ok", "bbox": None, "from_tight_crop": True}
        else:
            selfie_face, selfie_meta_face = extract_single_face_np(sf_arr, detector_backend=detector)

        if doc_face is None or selfie_face is None:
            # Falla de extracción de cara en doc o selfie (caso extremo)
            reason = {
                "doc": ("doc_face_" + doc_meta_face.get("reason", "fail")) if doc_face is None else "ok",
                "selfie": ("selfie_face_" + selfie_meta_face.get("reason", "fail")) if selfie_face is None else "ok",
            }
            return {
                "ok": False,
                "doc_ok": True,
                "selfie_ok": True,
                "reason": reason,
                "selfie_card_like": selfie_card_like,
                "card_meta": card_meta,
                "loose_meta": loose_meta,
                "strict_meta": strict_meta,
                "doc_face_reason": reason["doc"],
                "doc_mode_used": ("auto_failed" if (doc_mode == "auto" and not doc_ok) else doc_mode),
                "relax": bool(ignore_ar),
                "use_center_crop": bool(use_center_crop),
                "version": APP_VERSION,
                "selfie_meta": s_meta,
                "compare_source": "face_crops",
                "doc_face_bbox": doc_meta_face.get("bbox") if isinstance(doc_meta_face, dict) else None,
                "selfie_face_bbox": selfie_meta_face.get("bbox") if isinstance(selfie_meta_face, dict) else None,
            }

        # 3.c comparar SOLO las caras (saltando re-detección)
        vr = DeepFace.verify(
            img1_path=doc_face,
            img2_path=selfie_face,
            model_name="ArcFace",
            detector_backend="skip",     # clave: no re-detecta
            distance_metric="cosine",
            enforce_detection=False
        )
        dist = float(vr.get("distance", 1.0))
        verified = dist <= float(threshold)

        return {
            "ok": True,
            "verified": bool(verified),
            "distance": dist,
            "threshold": float(threshold),
            "model": "ArcFace",
            "detector": detector,
            "enforce": bool(enforce),
            "fallback_used": False,
            "note": "",

            "doc_ok": True,
            "selfie_ok": True,
            "card_meta": card_meta,
            "loose_meta": loose_meta,
            "strict_meta": strict_meta,
            "doc_mode_used": ("auto_failed" if (doc_mode == "auto" and not doc_ok) else doc_mode),
            "relax": bool(ignore_ar),
            "use_center_crop": bool(use_center_crop),
            "version": APP_VERSION,
            "selfie_meta": {"reason": "ok", "area_ratio": area_ratio, "tight_crop": bool(auto_tight_crop)},

            # nuevos campos de debug de comparación
            "compare_source": "face_crops",
            "doc_face_bbox": doc_meta_face.get("bbox"),
            "selfie_face_bbox": selfie_meta_face.get("bbox"),
        }

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": f"{type(e).__name__}: {str(e)}", "version": APP_VERSION}
        )

# =========================
# Uvicorn
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
