from paddleocr import PaddleOCR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import base64
import os
import tempfile
import cv2
import numpy as np
import re
import warnings

CUSTOM_CACHE_DIR = "C:/TempfileForDoctr"
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

# Set cache env vars
os.environ["MPLCONFIGDIR"] = CUSTOM_CACHE_DIR
os.environ["DOCTR_CACHE_DIR"] = CUSTOM_CACHE_DIR

# PaddleOCR model paths
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    show_log=False,
    det_model_dir="C:/my_paddleocr_models/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer",
    rec_model_dir="C:/my_paddleocr_models/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer",
    cls_model_dir="C:/my_paddleocr_models/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"
)

# Load doctr model
predictor = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
warnings.filterwarnings("ignore", category=UserWarning, module="paddle.utils.cpp_extension")

# Compile regex patterns once
pattern_name = re.compile(r'Name[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE)
pattern_date = re.compile(r'\d{2}[\s.\n-]*?[A-Za-z]{3}[\s.\n-]*?\d{4}')
pattern_num = re.compile(r'((?:\d[\s.\n-]*){17}|(?:\d[\s.\n-]*){13}|(?:\d[\s.\n-]*){10})')

def get_dcotr_ocr(image_input):
    if isinstance(image_input, str):
        doc = DocumentFile.from_images(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("image_input is an empty array")
        temp_path = os.path.join(CUSTOM_CACHE_DIR, f"temp_image_{os.urandom(8).hex()}.png")
        try:
            cv2.imwrite(temp_path, image_input)
            doc = DocumentFile.from_images(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        raise TypeError("Unsupported image input type.")

    result = predictor(doc)
    raw_text = []
    for page in result.pages:
        for block in page.blocks:
            for artefact in block.artefacts or []:
                raw_text.append(f"Table cell: {artefact.value}\n")
            for line in block.lines:
                raw_text.append(" ".join(word.value for word in line.words) + "\n")
    return "".join(raw_text).strip()

def get_paddle_ocr(image):
    results = ocr.ocr(image, cls=True)
    return "\n".join(line[1][0] for line in results[0]) if results and results[0] else ""

def process_raw_data(text):
    extracted = {}
    text_without_dates = text

    # Name
    matches_name = pattern_name.findall(text)
    name_paddle = re.sub(r"[^A-Za-z\s.\-\u0980-\u09FF]", "", matches_name[0]).strip() if matches_name else ""

    # DOB
    matches_date = pattern_date.findall(text)
    dob_paddle = None
    if matches_date:
        raw_date = re.sub(r'[\s.\n-]+', '', matches_date[0])
        try:
            date_obj = datetime.strptime(raw_date, "%d%b%Y")
            dob_paddle = date_obj.strftime("%Y-%m-%d")
        except Exception:
            pass
        text_without_dates = pattern_date.sub('', text)

    # NID
    matches_num = pattern_num.findall(text_without_dates)
    nid_paddle = re.sub(r'[\s.\n-]+', '', matches_num[0]) if matches_num else None

    extracted['Name'] = name_paddle
    extracted['DateOfBirth'] = dob_paddle
    extracted['IDNO'] = nid_paddle

    return extracted

def process_image(image_path):
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise ValueError(f"Failed to load image at {image_path}")

    # OCR from Paddle
    paddle_text1 = get_paddle_ocr(img_cv2)
    paddle_result = process_raw_data(paddle_text1)

    # OCR from Doctr
    doctr_text1 = get_dcotr_ocr(image_path)
    docrt_result = process_raw_data(doctr_text1)

    # Final merge
    final_name = {}
    if paddle_result['Name'].replace(" ", "") == docrt_result['Name'].replace(" ", ""):
        final_name['Name'] = max(paddle_result['Name'], docrt_result['Name'], key=len)
    else:
        final_name['Name'] = max((paddle_result['Name'], docrt_result['Name']), key=len)

    final_name['DateOfBirth'] = paddle_result['DateOfBirth'] or docrt_result['DateOfBirth'] or "Not Found"
    final_name['IDNO'] = paddle_result['IDNO'] or docrt_result['IDNO'] or "Not Found"

    final_name.update({'পিতা': "", 'মাতা': "", 'স্বামী': "", 'স্ত্রী': "", 'নাম': ""})
    return final_name

# FastAPI setup
app = FastAPI()

class ImageData(BaseModel):
    image_base64: str

@app.post("/extract-fields/")
async def extract_fields(data: ImageData):
    try:
        image_bytes = base64.b64decode(data.image_base64)
        with tempfile.NamedTemporaryFile(dir=CUSTOM_CACHE_DIR, suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        result = process_image(temp_path)
        os.unlink(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
