from paddleocr import PaddleOCR
import base64
import os
import tempfile
import cv2
import numpy as np
import re
from math import atan, degrees, radians, sin, cos, fabs
import warnings
from datetime import datetime
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel




CUSTOM_CACHE_DIR = "C:/TempfileForDoctr"

det_model_dir_path = "C:/my_paddleocr_models/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer"
rec_model_dir_path = "C:/my_paddleocr_models/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer"
cls_model_dir_path = "C:/my_paddleocr_models/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"

# Create custom cache directory if it doesn't exist
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

# Set environment variables for Matplotlib and DocTR
os.environ["MPLCONFIGDIR"] = CUSTOM_CACHE_DIR
os.environ["DOCTR_CACHE_DIR"] = CUSTOM_CACHE_DIR

# Initialize PaddleOCR with these precise model directory paths
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en', # Keep lang='en' as your det and rec models are English
    use_gpu=False,
    show_log=False,
    det_model_dir=det_model_dir_path,
    rec_model_dir=rec_model_dir_path,
    cls_model_dir=cls_model_dir_path
)

# model load doctr
predictor = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
warnings.filterwarnings("ignore", category=UserWarning, module="paddle.utils.cpp_extension")

def get_dcotr_ocr(image_input):
    # Ensure custom directory exists
    os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        doc = DocumentFile.from_images(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("image_input is an empty array")
        temp_file_name = f"temp_image_{os.urandom(8).hex()}.png"
        temp_file_path = os.path.join(CUSTOM_CACHE_DIR, temp_file_name)
        try:
            cv2.imwrite(temp_file_path, image_input)
            doc = DocumentFile.from_images(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        raise TypeError("Unsupported image input type. Must be a file path (str) or a NumPy array.")

    result = predictor(doc)
    raw_text = []
    for page in result.pages:
        for block in page.blocks:
            if block.artefacts:
                for artefact in block.artefacts:
                    raw_text.append(f"Table cell: {artefact.value}\n")
            for line in block.lines:
                raw_text.append(" ".join([word.value for word in line.words]) + "\n")
    return "".join(raw_text).strip()


def get_paddle_ocr(image):
    results = ocr.ocr(image, cls=True)
    all_text = ""
    for line in results[0]:
        text = line[1][0]
        all_text += text + "\n"
    #all_text=' '.join([line[1][0] for block in results for line in block]) if results and results[0] else ""
    return all_text



def process_raw_data(text):
    text_without_dates=text
    extracted = {}
    name_paddle = ''
    dob_paddle = ''
    nid_paddle = ''
    # Pattern for DDMonYYYY and DD Mon YYYY formats with optional multiple separators and newlines
    pattern_name = re.compile(r'Name[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE)
    pattern_date = re.compile(r'\d{2}[\s.\n-]*?[A-Za-z]{3}[\s.\n-]*?\d{4}')
    pattern_num = re.compile(r'((?:\d[\s.\n-]*){10}|(?:\d[\s.\n-]*){13}|(?:\d[\s.\n-]*){17})')

    # Find all date matches
    matches_name = pattern_name.findall(text)
    matches_date = pattern_date.findall(text)

    if matches_name:
        name_paddle= re.sub(r"[^A-Za-z\s\.\-\u0980-\u09FF]", "", matches_name[0]).strip()

        # print("Name:", name_paddle)
    else:
        name_paddle = ""

    if matches_date:
        # DOB = matches_date[0]
        dob_paddle = re.sub(r'[\s.\n-]+', '', matches_date[0])
        #dob_paddle = clean_date_of_birth(dob_paddle)
        date_obj = datetime.strptime(dob_paddle, "%d%b%Y")
        # Convert to YYYY-MM-DD
        dob_paddle = date_obj.strftime("%Y-%m-%d")
        #dob_paddle= clean_date_of_birth(dob_paddle)


        # print("DOB:", dob_paddle)

        # Remove dates from text
        text_without_dates = pattern_date.sub('', text)
        # print("text_without_dates:", text_without_dates
    else:
        dob_paddle = None


    # Pattern for 10, 13, 17 digit numbers with optional spaces, dashes, dots, newlines

    pattern_num = re.compile(r'\b((?:\d[\s-]){10}|(?:\d[\s-]){13}|(?:\d[\s-]*){17})\b')
    #pattern_num = re.compile(r'((?:\d[\s.\n-]*){10}|(?:\d[\s.\n-]*){13}|(?:\d[\s.\n-]*){17})')
    matches_num = pattern_num.findall(text_without_dates)
    print("pattern_num", matches_num)

    if matches_num:
        # Clean first match by removing all spaces, dashes, dots, newlines
        nid_paddle = re.sub(r'[\s.\n-]+', '', matches_num[0])
        # print("First Extracted Number:", nid_paddle)
    else:
        nid_paddle = None
        # print("No valid number found")

    extracted['Name'] = name_paddle
    # extracted['DateOfBirth'] = clean_date_of_birth(dob_paddle)
    extracted['DateOfBirth'] = dob_paddle
    extracted['IDNO'] = nid_paddle

    return extracted





app = FastAPI()
class ImageData(BaseModel):
    image_base64: str

def process_image(image_path):
    #image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/06-30-2025/3.png"
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise ValueError(f"Failed to load image at {image_path}")

    # for paddle process
    #paddle_text1 = get_paddle_ocr(img_cv2)
    #print(paddle_text1)
    #paddle_result = process_raw_data(paddle_text1)
    paddle_result = process_raw_data("nid 123456789")
    print('paddle prccess data: ', paddle_result)

    # for dcotr process
    #doctr_text1 = get_dcotr_ocr(image_path)
    #print(doctr_text1)
    docrt_result = process_raw_data("nid 123456789")
    #docrt_result = process_raw_data(doctr_text1)
    print('doctr prccess data: ', docrt_result)

    final_name = {}
    if paddle_result['Name'].replace(" ", "") == docrt_result['Name'].replace(" ", ""):

        if len(paddle_result['Name']) > len(docrt_result['Name']):
            final_name['Name'] = paddle_result['Name']
        else:
            final_name['Name'] = docrt_result['Name']
    else:
        if len(paddle_result['Name']) == len(docrt_result['Name']):
            final_name['Name'] = paddle_result['Name']
        elif len(paddle_result['Name']) > len(docrt_result['Name']):
            final_name['Name'] = paddle_result['Name']
        else:
            final_name['Name'] = docrt_result['Name']

    final_name['DateOfBirth'] = paddle_result['DateOfBirth'] or docrt_result['DateOfBirth'] or "Not Found"
    final_name['IDNO'] = paddle_result['IDNO'] or docrt_result['IDNO'] or "Not Found"
    final_name['পিতা']=""
    final_name['মাতা']=""
    final_name['স্বামী']=""
    final_name['স্ত্রী']=""
    final_name['নাম']=""



    return final_name

image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/06-30-2025/11.png"
result = process_image(image_path)
print(result)



# @app.post("/extract-fields/")
# async def extract_fields(data: ImageData):
#     try:
#         # Decode base64 to image
#         image_bytes = base64.b64decode(data.image_base64)
#
#         # Save to temporary file
#         with tempfile.NamedTemporaryFile(dir="C:/TempfileForDoctr", suffix=".png", delete=False) as temp_file:
#             temp_file.write(image_bytes)
#             temp_file_path = temp_file.name
#
#         # Process the image with the custom processing function
#         result = process_image(temp_file_path)
#
#         # Delete the temporary file
#         os.unlink(temp_file_path)
#
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))