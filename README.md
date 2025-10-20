# Face Verify API (FastAPI + ArcFace on DeepFace)

Servicio HTTP para verificar si la selfie coincide con la foto del documento (frontal).
- Modelo: **ArcFace** (métrica coseno; 0 = idéntico)
- Detector: **RetinaFace**
- Endpoint principal: `POST /verify`

---

## 🚀 Deploy en Railway (Nixpacks o Docker)

**Opción 1 — Nixpacks (sin Dockerfile):**
1. Crea un repo con estos archivos.
2. En Railway, "New Project" → "Deploy from GitHub".
3. Variables opcionales:
   - `ARC_THRESHOLD` (por defecto `0.38`)
   - `PORT` (`8080` por defecto, Railway la inyecta)
4. Railway detectará FastAPI/uvicorn y arrancará.

**Opción 2 — Dockerfile (incluido):**
1. Activa la opción de construir con Dockerfile.
2. Deploy y listo.

---

## 📦 Requisitos locales

```bash
python -m venv .venv
. .venv/bin/activate  # (Windows PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Probar
```bash
curl -X GET http://localhost:8080/health
curl -X POST "http://localhost:8080/verify" \
  -F "id_image=@/ruta/front_id.jpg" \
  -F "selfie=@/ruta/selfie.jpg" \
  -F "threshold=0.38"
```

Respuesta esperada:
```json
{
  "ok": true,
  "verified": true,
  "distance": 0.31,
  "threshold": 0.38,
  "model": "ArcFace",
  "detector": "retinaface"
}
```

---

## 🔧 Endpoint

`POST /verify` (multipart/form-data)
- **id_image**: archivo (imagen del documento frontal)
- **selfie**: archivo (selfie)
- **threshold**: float opcional (por defecto 0.38)

### Healthcheck
`GET /health`

---

## 🔐 Notas y Tips
- ArcFace con umbral 0.35–0.45 suele funcionar bien; empieza por 0.38.
- Mantén resolución decente (p.ej. 720p) para mejores embeddings.
- Este servicio **no** guarda archivos; procesa en memoria y responde JSON.
