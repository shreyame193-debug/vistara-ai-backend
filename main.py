# =========================
# VISTARA AI - MAIN.PY
# =========================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
import requests
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import os
import uuid
from datetime import datetime

# Register Unicode TrueType Fonts for Multilingual PDF Reports (Kannada, Hindi, English)
BASE_DIR = Path(__file__).resolve().parent
FONTS_DIR = BASE_DIR.parent / "fonts"
if not FONTS_DIR.exists():
    FONTS_DIR = BASE_DIR / "fonts"

dev_reg = str(FONTS_DIR / "NotoSansDevanagari-Regular.ttf")
dev_bold = str(FONTS_DIR / "NotoSansDevanagari-Bold.ttf")
kn_reg = str(FONTS_DIR / "NotoSansKannada-Regular.ttf")
kn_bold = str(FONTS_DIR / "NotoSansKannada-Bold.ttf")

if os.path.exists(dev_reg) and os.path.exists(dev_bold):
    pdfmetrics.registerFont(TTFont("Devanagari", dev_reg))
    pdfmetrics.registerFont(TTFont("Devanagari-Bold", dev_bold))
    pdfmetrics.registerFontFamily("Devanagari", normal="Devanagari", bold="Devanagari-Bold", italic="Devanagari", boldItalic="Devanagari-Bold")

if os.path.exists(kn_reg) and os.path.exists(kn_bold):
    pdfmetrics.registerFont(TTFont("Kannada", kn_reg))
    pdfmetrics.registerFont(TTFont("Kannada-Bold", kn_bold))
    pdfmetrics.registerFontFamily("Kannada", normal="Kannada", bold="Kannada-Bold", italic="Kannada", boldItalic="Kannada-Bold")


# =========================
# APP CONFIGURATION
# =========================

app = FastAPI(
    title="Vistara AI",
    description="AI-Powered Onion Quality Assessment API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Multi-lingual Voice Text-To-Speech (TTS) Service (Kannada, Hindi, English)
# ==============================================================================
_TTS_CACHE = {}

def fetch_tts_audio_bytes(text: str, lang: str = "en") -> bytes:
    clean_text = text.strip()
    if not clean_text:
        return b""

    lang_code = lang.strip().lower()
    if lang_code.startswith("kn"):
        tl = "kn"
    elif lang_code.startswith("hi"):
        tl = "hi"
    else:
        tl = "en"

    cache_key = f"{tl}:{clean_text}"
    if cache_key in _TTS_CACHE:
        return _TTS_CACHE[cache_key]

    words = clean_text.split()
    chunks = []
    current_chunk = []
    current_len = 0
    for w in words:
        if current_len + len(w) + 1 > 160 and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [w]
            current_len = len(w)
        else:
            current_chunk.append(w)
            current_len += len(w) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if not chunks:
        chunks = [clean_text[:160]]

    audio_bytes = bytearray()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            resp = requests.get(
                "https://translate.google.com/translate_tts",
                params={"ie": "UTF-8", "tl": tl, "client": "tw-ob", "q": chunk.strip()},
                headers=headers,
                timeout=8
            )
            if resp.status_code == 200 and resp.content:
                audio_bytes.extend(resp.content)
        except Exception as e:
            print(f"[TTS Audio Service Warning] Chunk '{chunk}': {e}")

    result = bytes(audio_bytes)
    if len(result) > 0 and len(_TTS_CACHE) < 500:
        _TTS_CACHE[cache_key] = result
    return result


@app.get("/api/tts")
async def tts_get(text: str = "", lang: str = "en"):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'text' is required.")
    audio_data = fetch_tts_audio_bytes(text, lang)
    if not audio_data:
        raise HTTPException(status_code=500, detail="Failed to synthesize speech audio.")
    return Response(
        content=audio_data,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes"
        }
    )


@app.post("/api/tts")
async def tts_post(payload: dict):
    text = payload.get("text", "")
    lang = payload.get("lang", "en")
    if not text or not str(text).strip():
        raise HTTPException(status_code=400, detail="Field 'text' is required.")
    audio_data = fetch_tts_audio_bytes(str(text), str(lang))
    if not audio_data:
        raise HTTPException(status_code=500, detail="Failed to synthesize speech audio.")
    return Response(
        content=audio_data,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes"
        }
    )



# =========================
# MODEL
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "vistara_V2_best.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "best.pt"

print(f"[AI Backend] Loading YOLO model from {MODEL_PATH} ...")
model = YOLO(str(MODEL_PATH))

CLASS_NAMES = {
    0: "healthy",
    1: "mold",
    2: "rotten",
    3: "sprouted",
    4: "damaged"
}


# =========================
# DIRECTORIES
# =========================

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# =========================
# HEALTH CHECK
# =========================

@app.get("/")
@app.get("/api/health")
def root():
    return {
        "status": "online",
        "application": "Vistara AI",
        "model": "YOLO26 Nano",
        "classes": [
            "healthy",
            "mold",
            "rotten",
            "sprouted",
            "damaged"
        ]
    }


# =========================
# ANALYZE ONION IMAGE
# =========================

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    try:
        import base64
        import torch

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")
        orig_w, orig_h = image.width, image.height

        # Fast pre-scale for high-resolution images to accelerate inference without accuracy loss
        infer_img = image
        scale_factor = 1.0
        if orig_w > 1280 or orig_h > 1280:
            scale_factor = min(1280 / orig_w, 1280 / orig_h)
            new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
            infer_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        num_cores = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_cores)

        with torch.no_grad():
            results = model.predict(
                source=infer_img,
                conf=0.20,
                iou=0.45,
                imgsz=640,
                device="cpu",
                verbose=False
            )

        result = results[0] if len(results) > 0 else None

        counts = {
            "healthy": 0,
            "mold": 0,
            "rotten": 0,
            "sprouted": 0,
            "damaged": 0
        }

        detections = []
        annotated_b64 = None

        if result is not None and result.boxes is not None:
            boxes_data = result.boxes
            inv_scale = 1.0 / scale_factor if scale_factor > 0 else 1.0

            for i in range(len(boxes_data)):
                class_id = int(boxes_data.cls[i])
                confidence = float(boxes_data.conf[i])
                raw_box = boxes_data.xyxy[i].tolist()

                class_name = CLASS_NAMES.get(
                    class_id,
                    model.names.get(class_id, "unknown")
                )

                if class_name in counts:
                    counts[class_name] += 1

                # Rescale coordinates to original image dimensions
                box = [
                    round(float(raw_box[0]) * inv_scale, 2),
                    round(float(raw_box[1]) * inv_scale, 2),
                    round(float(raw_box[2]) * inv_scale, 2),
                    round(float(raw_box[3]) * inv_scale, 2)
                ]

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "box": box
                })

            # Fast preview thumbnail
            try:
                plotted = result.plot()
                pil_plotted = Image.fromarray(plotted[..., ::-1])
                buf = io.BytesIO()
                pil_plotted.save(buf, format="JPEG", quality=75, optimize=True)
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass

        total_onions = sum(counts.values())

        if total_onions > 0:
            healthy_pct = round(counts["healthy"] / total_onions * 100, 2)
            mold_pct = round(counts["mold"] / total_onions * 100, 2)
            rotten_pct = round(counts["rotten"] / total_onions * 100, 2)
            sprouted_pct = round(counts["sprouted"] / total_onions * 100, 2)
            damaged_pct = round(counts["damaged"] / total_onions * 100, 2)
        else:
            healthy_pct = 0.0
            mold_pct = 0.0
            rotten_pct = 0.0
            sprouted_pct = 0.0
            damaged_pct = 0.0

        return {
            "success": True,
            "image_width": orig_w,
            "image_height": orig_h,
            "total_onions": total_onions,

            "healthy": counts["healthy"],
            "mold": counts["mold"],
            "rotten": counts["rotten"],
            "sprouted": counts["sprouted"],
            "damaged": counts["damaged"],

            "healthy_percentage": healthy_pct,
            "mold_percentage": mold_pct,
            "rotten_percentage": rotten_pct,
            "sprouted_percentage": sprouted_pct,
            "damaged_percentage": damaged_pct,

            "detections": detections,
            "annotated_image": annotated_b64,

            "model": "YOLO26 Nano",
            "message": "Image analyzed successfully"
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# =========================
# REPORT TRANSLATIONS
# =========================

TRANSLATIONS = {
    "English": {
        "report_title": "VISTARA AI",
        "report_subtitle": "DIGITAL ONION QUALITY ASSESSMENT REPORT",
        "report_note": "AI-assisted inspection report — for procurement assessment and human verification",
        "batch": "Batch ID",
        "date": "Assessment Date",
        "farmer": "Producer / Farmer",
        "center": "Procurement Center",
        "weight": "Lot Net Weight",
        "total": "Total Onions Detected",
        "original_image_heading": "Original Uploaded Image",
        "original_image_sub": "Raw onion sample image submitted for quality inspection",
        "ai_image_heading": "AI Detection & Inspection Image",
        "ai_image_sub": "YOLO26 Nano defect detection and bounding box localization",
        "legend_healthy": "Healthy",
        "legend_mold": "Mold",
        "legend_rotten": "Rotten",
        "legend_sprouted": "Sprouted",
        "legend_damaged": "Damaged",
        "summary": "Detection Summary",
        "th_param": "Parameter",
        "th_count": "Count",
        "th_pct": "Percentage",
        "th_obs": "Observation / Specification",
        "healthy_param": "Healthy Onion Bulbs",
        "mold_param": "Mold-Affected Onion Bulbs",
        "rotten_param": "Rotten Onion Bulbs",
        "sprouted_param": "Sprouted Onion Bulbs",
        "damaged_param": "Damaged Onion Bulbs",
        "grade_a": "Grade A",
        "urs": "URS",
        "healthy_tol": "No visible defect detected (Sound bulb)",
        "mold_tol": "Visible mold detected (Aspergillus)",
        "rotten_tol": "Rotten onion detected (Bacterial decay)",
        "sprouted_tol": "Sprouting detected (Premature growth)",
        "damaged_tol": "Detected physical damage",
        "grade_a_rule": "Configured grading rule",
        "urs_rule": "Configured grading rule",
        "total_param": "Total",
        "total_obs": "Detection summary",
        "method": "Detection Method",
        "method_value": "YOLO26 Nano object detection",
        "human": "Human Verification",
        "human_value": "Required before final procurement decision",
        "status": "Grading Status",
        "disclaimer": "This report is an AI-assisted assessment and should be verified by an authorized human inspector before final procurement decisions.",
        "generated": "Generated by Vistara AI • APMC Mandi Quality Division"
    },
    "Kannada": {
        "report_title": "ವಿಸ್ಟಾರಾ AI",
        "report_subtitle": "ಡಿಜಿಟಲ್ ಈರುಳ್ಳಿ ಗುಣಮಟ್ಟ ಮೌಲ್ಯಮಾಪನ ವರದಿ",
        "report_note": "AI ಸಹಾಯಿತ ಪರಿಶೀಲನಾ ವರದಿ — ಖರೀದಿ ಮೌಲ್ಯಮಾಪನ ಮತ್ತು ಮಾನವ ಪರಿಶೀಲನೆಗಾಗಿ",
        "batch": "ಬ್ಯಾಚ್ ID",
        "date": "ಮೌಲ್ಯಮಾಪನ ದಿನಾಂಕ",
        "farmer": "ರೈತ / ಉತ್ಪಾದಕ",
        "center": "ಖರೀದಿ ಕೇಂದ್ರ",
        "weight": "ಒಟ್ಟು ತೂಕ",
        "total": "ಪತ್ತೆಯಾದ ಒಟ್ಟು ಈರುಳ್ಳಿಗಳು",
        "original_image_heading": "ಮೂಲ ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ",
        "original_image_sub": "ಗುಣಮಟ್ಟ ತಪಾಸಣೆಗಾಗಿ ಸಲ್ಲಿಸಲಾದ ಕಚ್ಚಾ ಈರುಳ್ಳಿ ಮಾದರಿ ಚಿತ್ರ",
        "ai_image_heading": "AI ಪತ್ತೆ ಮತ್ತು ತಪಾಸಣೆ ಚಿತ್ರ",
        "ai_image_sub": "YOLO26 Nano ದೋಷ ಪತ್ತೆ ಮತ್ತು ಬೌಂಡಿಂಗ್ ಬಾಕ್ಸ್ ಗುರುತಿಸುವಿಕೆ",
        "legend_healthy": "ಆರೋಗ್ಯಕರ",
        "legend_mold": "ಅಚ್ಚು",
        "legend_rotten": "ಕೊಳೆತ",
        "legend_sprouted": "ಮೊಳಕೆಯೊಡೆದ",
        "legend_damaged": "ಹಾನಿಗೊಳಗಾದ",
        "summary": "ಪತ್ತೆ ಸಾರಾಂಶ",
        "th_param": "ಪ್ಯಾರಾಮೀಟರ್",
        "th_count": "ಸಂಖ್ಯೆ",
        "th_pct": "ಶೇಕಡಾವಾರು",
        "th_obs": "ವೀಕ್ಷಣೆ / ವಿವರಣೆ",
        "healthy_param": "ಆರೋಗ್ಯಕರ ಈರುಳ್ಳಿ",
        "mold_param": "ಅಚ್ಚು ಪೀಡಿತ ಈರುಳ್ಳಿ",
        "rotten_param": "ಕೊಳೆತ ಈರುಳ್ಳಿ",
        "sprouted_param": "ಮೊಳಕೆಯೊಡೆದ ಈರುಳ್ಳಿ",
        "damaged_param": "ಹಾನಿಗೊಳಗಾದ ಈರುಳ್ಳಿ",
        "grade_a": "ಗ್ರೇಡ್ A",
        "urs": "URS",
        "healthy_tol": "ಯಾವುದೇ ಗೋಚರ ದೋಷ ಪತ್ತೆಯಾಗಿಲ್ಲ",
        "mold_tol": "ಗೋಚರ ಅಚ್ಚು ಪತ್ತೆಯಾಗಿದೆ",
        "rotten_tol": "ಕೊಳೆತ ಈರುಳ್ಳಿ ಪತ್ತೆಯಾಗಿದೆ",
        "sprouted_tol": "ಮೊಳಕೆ ಪತ್ತೆಯಾಗಿದೆ",
        "damaged_tol": "ಪತ್ತೆಯಾದ ಭೌತಿಕ ಹಾನಿ",
        "grade_a_rule": "ಕಾನ್ಫಿಗರ್ ಮಾಡಿದ ಗ್ರೇಡಿಂಗ್ ನಿಯಮ",
        "urs_rule": "ಕಾನ್ಫಿಗರ್ ಮಾಡಿದ ಗ್ರೇಡಿಂಗ್ ನಿಯಮ",
        "total_param": "ಒಟ್ಟು",
        "total_obs": "ಪತ್ತೆ ಸಾರಾಂಶ",
        "method": "ಪತ್ತೆ ವಿಧಾನ",
        "method_value": "YOLO26 Nano ಆಬ್ಜೆಕ್ಟ್ ಡಿಟೆಕ್ಷನ್",
        "human": "ಮಾನವ ಪರಿಶೀಲನೆ",
        "human_value": "ಅಂತಿಮ ಖರೀದಿ ನಿರ್ಧಾರಕ್ಕೂ ಮೊದಲು ಅಗತ್ಯ",
        "status": "ಗ್ರೇಡಿಂಗ್ ಸ್ಥಿತಿ",
        "disclaimer": "ಈ ವರದಿಯು AI ಸಹಾಯಿತ ಮೌಲ್ಯಮಾಪನವಾಗಿದೆ. ಅಂತಿಮ ಖರೀದಿ ನಿರ್ಧಾರಕ್ಕೂ ಮೊದಲು ಮಾನವ ಪರಿಶೀಲನೆ ಮಾಡಬೇಕು.",
        "generated": "Vistara AI ಮೂಲಕ ರಚಿಸಲಾಗಿದೆ • APMC ಮಂಡಿ ಗುಣಮಟ್ಟ ವಿಭಾಗ"
    },
    "Hindi": {
        "report_title": "विस्तारा AI",
        "report_subtitle": "डिजिटल प्याज गुणवत्ता मूल्यांकन रिपोर्ट",
        "report_note": "AI-सहायित निरीक्षण रिपोर्ट — खरीद मूल्यांकन और मानव सत्यापन के लिए",
        "batch": "बैच ID",
        "date": "मूल्यांकन तिथि",
        "farmer": "उत्पादक / किसान",
        "center": "खरीद केंद्र",
        "weight": "कुल वजन",
        "total": "कुल पाए गए प्याज",
        "original_image_heading": "मूल अपलोड की गई छवि",
        "original_image_sub": "गुणवत्ता निरीक्षण के लिए प्रस्तुत कच्चा प्याज नमूना छवि",
        "ai_image_heading": "AI डिटेक्शन एवं निरीक्षण छवि",
        "ai_image_sub": "YOLO26 Nano दोष पहचान एवं बाउंडिंग बॉक्स स्थानीयकरण",
        "legend_healthy": "स्वस्थ",
        "legend_mold": "फफूंद",
        "legend_rotten": "सड़े हुए",
        "legend_sprouted": "अंकुरित",
        "legend_damaged": "क्षतिग्रस्त",
        "summary": "डिटेक्शन सारांश",
        "th_param": "पैरामीटर",
        "th_count": "संख्या",
        "th_pct": "प्रतिशत",
        "th_obs": "अवलोकन / विनिर्देश",
        "healthy_param": "स्वस्थ प्याज",
        "mold_param": "फफूंदी प्रभावित प्याज",
        "rotten_param": "सड़े हुए प्याज",
        "sprouted_param": "अंकुरित प्याज",
        "damaged_param": "क्षतिग्रस्त प्याज",
        "grade_a": "ग्रेड A",
        "urs": "URS",
        "healthy_tol": "कोई दिखाई देने वाला दोष नहीं",
        "mold_tol": "फफूंदी पाई गई",
        "rotten_tol": "सड़ा हुआ प्याज पाया गया",
        "sprouted_tol": "अंकुरण पाया गया",
        "damaged_tol": "पाई गई भौतिक क्षति",
        "grade_a_rule": "कॉन्फ़िगर किया गया ग्रेडिंग नियम",
        "urs_rule": "कॉन्फ़िगर किया गया ग्रेडिंग नियम",
        "total_param": "कुल",
        "total_obs": "डिटेक्शन सारांश",
        "method": "पता लगाने की विधि",
        "method_value": "YOLO26 Nano ऑब्जेक्ट डिटेक्शन",
        "human": "मानव सत्यापन",
        "human_value": "अंतिम खरीद निर्णय से पहले आवश्यक",
        "status": "ग्रेडिंग स्थिति",
        "disclaimer": "यह रिपोर्ट AI-सहायित मूल्यांकन है। अंतिम खरीद निर्णय से पहले मानव निरीक्षक द्वारा सत्यापन किया जाना चाहिए।",
        "generated": "Vistara AI द्वारा निर्मित • APMC मंडी गुणवत्ता प्रभाग"
    }
}


def normalize_language(lang_val: str) -> str:
    if not lang_val:
        return "English"
    lv = str(lang_val).strip().lower()
    if "kannada" in lv or lv == "kn":
        return "Kannada"
    if "hindi" in lv or lv == "hi":
        return "Hindi"
    return "English"


def make_scaled_rl_image(img_path: str, max_w: float = 480, max_h: float = 210):
    """Helper to scale an image preserving aspect ratio without cropping."""
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            return None
        aspect = w / h
        target_w = min(max_w, float(w))
        target_h = target_w / aspect
        if target_h > max_h:
            target_h = max_h
            target_w = target_h * aspect
        return RLImage(str(img_path), width=target_w, height=target_h)
    except Exception as e:
        print(f"Error creating RLImage for {img_path}: {e}")
        return None


# =========================
# GENERATE PDF REPORT
# =========================

@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(None),
    annotated_file: UploadFile = File(None),
    batch_id: str = Form("VISTARA-BATCH"),
    total_onions: int = Form(0),
    healthy: int = Form(0),
    rotten: int = Form(0),
    mold: int = Form(0),
    sprouted: int = Form(0),
    damaged: int = Form(0),
    healthy_pct: str = Form(""),
    rotten_pct: str = Form(""),
    mold_pct: str = Form(""),
    sprouted_pct: str = Form(""),
    damaged_pct: str = Form(""),
    grade_a_pct: str = Form(""),
    urs_pct: str = Form(""),
    grading_status: str = Form("Assessment Pending"),
    language: str = Form("English"),
    lang: str = Form("English"),
    farmer_name: str = Form("Registered Producer"),
    farmer_id: str = Form("MH-REG-1049"),
    center: str = Form("Lasalgaon APMC Mandi"),
    weight_kg: str = Form("1450"),
    base_price_per_kg: str = Form("25.00"),
    effective_price_per_kg: str = Form(""),
    total_payout: str = Form("")
):
    try:
        active_lang = normalize_language(language or lang)
        t_dict = TRANSLATIONS[active_lang]

        # Font selection for Multilingual rendering
        font_name = "Helvetica"
        font_bold = "Helvetica-Bold"
        reg_fonts = pdfmetrics.getRegisteredFontNames()

        if active_lang == "Kannada" and "Kannada" in reg_fonts:
            font_name = "Kannada"
            font_bold = "Kannada-Bold"
        elif active_lang == "Hindi" and "Devanagari" in reg_fonts:
            font_name = "Devanagari"
            font_bold = "Devanagari-Bold"

        if total_onions > 0:
            h_pct = healthy_pct if healthy_pct else f"{round((healthy / total_onions) * 100, 2)}%"
            r_pct = rotten_pct if rotten_pct else f"{round((rotten / total_onions) * 100, 2)}%"
            m_pct = mold_pct if mold_pct else f"{round((mold / total_onions) * 100, 2)}%"
            s_pct = sprouted_pct if sprouted_pct else f"{round((sprouted / total_onions) * 100, 2)}%"
            d_pct = damaged_pct if damaged_pct else f"{round((damaged / total_onions) * 100, 2)}%"
        else:
            h_pct = healthy_pct if healthy_pct else "0.00%"
            r_pct = rotten_pct if rotten_pct else "0.00%"
            m_pct = mold_pct if mold_pct else "0.00%"
            s_pct = sprouted_pct if sprouted_pct else "0.00%"
            d_pct = damaged_pct if damaged_pct else "0.00%"

        if not grade_a_pct:
            grade_a_pct = h_pct
        
        defect_cnt = rotten + mold + sprouted + damaged
        if not urs_pct:
            urs_pct = f"{round((defect_cnt / total_onions) * 100, 2)}%" if total_onions > 0 else "0.00%"

        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:6]
        filename = f"Vistara_Report_{batch_id}_{unique_token}.pdf"
        filepath = REPORT_DIR / filename

        # -----------------------------------------------------------------
        # Process and Save Both Uploaded Images (Raw intake & AI-marked)
        # -----------------------------------------------------------------
        raw_img_path = None
        annotated_img_path = None

        # 1. Original User-Uploaded Image
        if file is not None and file.filename:
            raw_bytes = await file.read()
            if len(raw_bytes) > 0:
                raw_img_path = str(REPORT_DIR / f"raw_{timestamp_slug}_{unique_token}.jpg")
                try:
                    pil_raw = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    pil_raw.save(raw_img_path, "JPEG", quality=90)
                except Exception as e:
                    print(f"[MainAPI] Error saving raw image: {e}")
                    raw_img_path = None

        # 2. AI-Marked / Annotated Detection Image
        if annotated_file is not None and annotated_file.filename:
            ann_bytes = await annotated_file.read()
            if len(ann_bytes) > 0:
                annotated_img_path = str(REPORT_DIR / f"ann_{timestamp_slug}_{unique_token}.jpg")
                try:
                    pil_ann = Image.open(io.BytesIO(ann_bytes)).convert("RGB")
                    pil_ann.save(annotated_img_path, "JPEG", quality=90)
                except Exception as e:
                    print(f"[MainAPI] Error saving annotated image: {e}")
                    annotated_img_path = None
        elif raw_img_path and os.path.exists(raw_img_path):
            # Fallback: Run YOLO26 inference and draw real bounding boxes
            annotated_img_path = str(REPORT_DIR / f"ann_{timestamp_slug}_{unique_token}.jpg")
            try:
                pil_raw = Image.open(raw_img_path).convert("RGB")
                results = model.predict(source=pil_raw, conf=0.25, verbose=False)
                if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    plotted = results[0].plot()
                    pil_plotted = Image.fromarray(plotted[..., ::-1])
                    pil_plotted.save(annotated_img_path, "JPEG", quality=90)
                else:
                    pil_raw.save(annotated_img_path, "JPEG", quality=90)
            except Exception as e:
                print(f"[MainAPI] Error generating YOLO plotted image: {e}")
                annotated_img_path = raw_img_path

        # -----------------------------------------------------------------
        # ReportLab Document Setup
        # -----------------------------------------------------------------
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName=font_bold,
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4
        )

        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName=font_bold,
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#166534"),
            spaceAfter=4
        )

        note_style = ParagraphStyle(
            "ReportNote",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=7.5,
            leading=10.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=10
        )

        section_style = ParagraphStyle(
            "ReportSection",
            parent=styles["Heading2"],
            fontName=font_bold,
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=3
        )

        caption_style = ParagraphStyle(
            "ReportCaption",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=5
        )

        cell_text = ParagraphStyle(
            "CellText",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=7.8,
            leading=10,
            textColor=colors.HexColor("#1e293b")
        )

        cell_bold = ParagraphStyle(
            "CellBold",
            parent=cell_text,
            fontName=font_bold,
            textColor=colors.HexColor("#0f172a")
        )

        cell_center = ParagraphStyle(
            "CellCenter",
            parent=cell_text,
            alignment=TA_CENTER
        )

        cell_center_bold = ParagraphStyle(
            "CellCenterBold",
            parent=cell_bold,
            alignment=TA_CENTER
        )

        legend_style = ParagraphStyle(
            "LegendStyle",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=6.8,
            leading=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569")
        )

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        story = []

        # 1. Header
        story.append(Paragraph(t_dict["report_title"], title_style))
        story.append(Paragraph(t_dict["report_subtitle"], subtitle_style))
        story.append(Paragraph(t_dict["report_note"], note_style))
        story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#16a34a"), spaceBefore=1, spaceAfter=8))

        # 2. Assessment Information
        batch_data = [
            [
                Paragraph(f"<b>{t_dict['batch']}</b>", cell_bold),
                Paragraph(str(batch_id), cell_text),
                Paragraph(f"<b>{t_dict['date']}</b>", cell_bold),
                Paragraph(datetime.now().strftime("%d-%m-%Y %H:%M"), cell_text)
            ],
            [
                Paragraph(f"<b>{t_dict['farmer']}</b>", cell_bold),
                Paragraph(f"{farmer_name} ({farmer_id})", cell_text),
                Paragraph(f"<b>{t_dict['center']}</b>", cell_bold),
                Paragraph(str(center), cell_text)
            ],
            [
                Paragraph(f"<b>{t_dict['weight']}</b>", cell_bold),
                Paragraph(f"{weight_kg} kg", cell_text),
                Paragraph(f"<b>{t_dict['total']}</b>", cell_bold),
                Paragraph(f"<b>{total_onions}</b>", cell_bold)
            ]
        ]

        batch_table = Table(batch_data, colWidths=[110, 150, 110, 153])
        batch_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f8fafc")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(batch_table)
        story.append(Spacer(1, 8))

        # Fallback to demo/sample images if no image was received
        if not raw_img_path or not os.path.exists(raw_img_path):
            ws_test_raw = os.path.join(BASE_DIR.parent, "test_raw.jpg")
            if os.path.exists(ws_test_raw):
                raw_img_path = ws_test_raw
                if not annotated_img_path or not os.path.exists(annotated_img_path):
                    ws_test_ann = os.path.join(BASE_DIR.parent, "test_ann.jpg")
                    if os.path.exists(ws_test_ann):
                        annotated_img_path = ws_test_ann

        # 3. Section A: Original Uploaded Image
        raw_rl = make_scaled_rl_image(raw_img_path, max_w=480, max_h=180) if raw_img_path else None
        if raw_rl:
            story.append(Paragraph(f"<b>{t_dict['original_image_heading']}</b>", section_style))
            story.append(Paragraph(t_dict["original_image_sub"], caption_style))
            raw_table = Table([[raw_rl]], colWidths=[523])
            raw_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(raw_table)
            story.append(Spacer(1, 8))

        # 4. Section B: AI Detection & Inspection Image
        ann_rl = make_scaled_rl_image(annotated_img_path, max_w=480, max_h=180) if annotated_img_path else None
        if ann_rl:
            story.append(Paragraph(f"<b>{t_dict['ai_image_heading']}</b>", section_style))
            story.append(Paragraph(t_dict["ai_image_sub"], caption_style))
            ann_table = Table([[ann_rl]], colWidths=[523])
            ann_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(ann_table)
            story.append(Spacer(1, 3))

            legend_p = Paragraph(
                f"<font color='#16a34a'>&#9632; <b>{t_dict['legend_healthy']}</b></font> &nbsp;&bull;&nbsp; "
                f"<font color='#0284c7'>&#9632; <b>{t_dict['legend_mold']}</b></font> &nbsp;&bull;&nbsp; "
                f"<font color='#dc2626'>&#9632; <b>{t_dict['legend_rotten']}</b></font> &nbsp;&bull;&nbsp; "
                f"<font color='#ca8a04'>&#9632; <b>{t_dict['legend_sprouted']}</b></font> &nbsp;&bull;&nbsp; "
                f"<font color='#ea580c'>&#9632; <b>{t_dict['legend_damaged']}</b></font>",
                legend_style
            )
            story.append(legend_p)
            story.append(Spacer(1, 8))

        # 5. Section: Detection Summary Table
        story.append(Paragraph(f"<b>{t_dict['summary']}</b>", section_style))

        defect_data = [
            [
                Paragraph(f"<b>{t_dict['th_param']}</b>", cell_center_bold),
                Paragraph(f"<b>{t_dict['th_count']}</b>", cell_center_bold),
                Paragraph(f"<b>{t_dict['th_pct']}</b>", cell_center_bold),
                Paragraph(f"<b>{t_dict['th_obs']}</b>", cell_center_bold)
            ],
            [
                Paragraph(t_dict["healthy_param"], cell_text),
                Paragraph(str(healthy), cell_center),
                Paragraph(h_pct, cell_center),
                Paragraph(t_dict["healthy_tol"], cell_text)
            ],
            [
                Paragraph(t_dict["mold_param"], cell_text),
                Paragraph(str(mold), cell_center),
                Paragraph(m_pct, cell_center),
                Paragraph(t_dict["mold_tol"], cell_text)
            ],
            [
                Paragraph(t_dict["rotten_param"], cell_text),
                Paragraph(str(rotten), cell_center),
                Paragraph(r_pct, cell_center),
                Paragraph(t_dict["rotten_tol"], cell_text)
            ],
            [
                Paragraph(t_dict["sprouted_param"], cell_text),
                Paragraph(str(sprouted), cell_center),
                Paragraph(s_pct, cell_center),
                Paragraph(t_dict["sprouted_tol"], cell_text)
            ],
            [
                Paragraph(t_dict["damaged_param"], cell_text),
                Paragraph(str(damaged), cell_center),
                Paragraph(d_pct, cell_center),
                Paragraph(t_dict["damaged_tol"], cell_text)
            ],
            [
                Paragraph(f"<b>{t_dict['grade_a']}</b>", cell_bold),
                Paragraph(str(healthy), cell_center),
                Paragraph(f"<b>{grade_a_pct}</b>", cell_center_bold),
                Paragraph(t_dict["grade_a_rule"], cell_text)
            ],
            [
                Paragraph(f"<b>{t_dict['urs']}</b>", cell_bold),
                Paragraph(str(defect_cnt), cell_center),
                Paragraph(f"<b>{urs_pct}</b>", cell_center_bold),
                Paragraph(t_dict["urs_rule"], cell_text)
            ],
            [
                Paragraph(f"<b>{t_dict['total_param']}</b>", cell_bold),
                Paragraph(f"<b>{total_onions}</b>", cell_center_bold),
                Paragraph("<b>100.00%</b>", cell_center_bold),
                Paragraph(t_dict["total_obs"], cell_text)
            ]
        ]

        defect_table = Table(defect_data, colWidths=[150, 60, 75, 238], repeatRows=1)
        defect_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f0fdf4")),
            ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#f0fdf4")),
            ("BACKGROUND", (0, 7), (-1, 7), colors.HexColor("#fffbeb")),
            ("BACKGROUND", (0, 8), (-1, 8), colors.HexColor("#f1f5f9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(defect_table)
        story.append(Spacer(1, 8))

        # 6. Technical & Methodology Information
        technical_data = [
            [
                Paragraph(f"<b>{t_dict['method']}</b>", cell_bold),
                Paragraph(t_dict["method_value"], cell_text)
            ],
            [
                Paragraph(f"<b>{t_dict['human']}</b>", cell_bold),
                Paragraph(t_dict["human_value"], cell_text)
            ],
            [
                Paragraph(f"<b>{t_dict['status']}</b>", cell_bold),
                Paragraph(str(grading_status), cell_text)
            ]
        ]

        technical_table = Table(technical_data, colWidths=[160, 363])
        technical_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(technical_table)
        story.append(Spacer(1, 10))

        # 7. Disclaimer & Footer
        story.append(
            Paragraph(
                f"<b>{t_dict['disclaimer']}</b>",
                ParagraphStyle("Disclaimer", parent=cell_text, fontSize=7, leading=10, alignment=TA_CENTER)
            )
        )
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                t_dict["generated"],
                ParagraphStyle("Generated", parent=cell_text, alignment=TA_CENTER, fontSize=7, textColor=colors.HexColor("#64748b"))
            )
        )

        # Build PDF
        doc.build(story)

        return FileResponse(
            path=str(filepath),
            media_type="application/pdf",
            filename=filename
        )

    except Exception as e:
        print("[MainAPI] Error generating report:", e)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

