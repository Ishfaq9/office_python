import cv2
import numpy as np
import os
from pathlib import Path
import tempfile
import pytesseract
import easyocr
import re
import json
import sys
from PIL import Image
from math import atan, degrees, radians, sin, cos, fabs
import warnings
from paddleocr import PaddleOCR
from datetime import datetime
import os
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor



# Define custom cache directory
CUSTOM_CACHE_DIR = "C:/TempfileForDoctr"

# Create custom cache directory if it doesn't exist
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

# Set environment variables for Matplotlib and DocTR
os.environ["MPLCONFIGDIR"] = CUSTOM_CACHE_DIR
os.environ["DOCTR_CACHE_DIR"] = CUSTOM_CACHE_DIR


predictor = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')



sys.stdout.reconfigure(encoding='utf-8')
# # Get image path from argument
if len(sys.argv) < 2:
    print(json.dumps({"error": "Image path is required"}))
    sys.exit(1)


# Tesseract path
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Suppress PaddleOCR warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="paddle.utils.cpp_extension")
# reader = easyocr.Reader(['en'],gpu=False, model_storage_directory =r'C:\easy\model',user_network_directory =r'C:\easy\network')
# # Initialize PaddleOCR globally
# ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

# Compile regex patterns for efficiency
fields_code1 = {
    'নাম': re.compile(r'নাম[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'Name': re.compile(r'Name[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'পিতা': re.compile(r'পিতা[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'মাতা': re.compile(r'মাতা[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'স্বামী': re.compile(r'স্বামী[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'স্ত্রী': re.compile(r'স্ত্রী[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'DateOfBirth': re.compile(r'Date of Birth[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    #'IDNO': re.compile(r'(?:ID\s*NO|NID\s*No\.?|ID|NIDNo|NID\s*NO|\|D NO|ID N|NID\s*No|ID\s*N0)\s*[:：]*\s*([\d\s]{8,30})', re.IGNORECASE | re.MULTILINE)
'IDNO': re.compile(r'\b(?:(?:[IN1l|]{1,2}D|NID)(?:\s*NO|\s*No|\s*N0|[-_]?NO|NO|No|no|\s*N|\s*N0|[-_]?N|[-_]?N0|\s*#)?|NID\s*(?:Number|Num|NUM)|\|N0|\|NO|\|N)\b[:\.]?\s*([\d\s-]{8,30})', re.IGNORECASE | re.MULTILINE)


}

fields_code2 = {
    'নাম': re.compile(r'নাম[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'Name': re.compile(r'Name[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'পিতা': re.compile(r'পিতা[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'মাতা': re.compile(r'মাতা[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE),
    'স্বামী': re.compile(r'(?:স্বামী|স্বা[:;মী-]*|husband|sami)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্ত্রী|Date|ID)', re.IGNORECASE | re.MULTILINE),
    'স্ত্রী': re.compile(r'(?:স্ত্রী|স্ত্র[:;ী-]*|wife|stri)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|Date|ID)', re.IGNORECASE | re.MULTILINE),
    'DateOfBirth': re.compile(r'(?:Date of Birth|Date ofBirth|DOB|Date|Birth)[:;\s-]*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|স্ত্রী|ID)', re.IGNORECASE | re.MULTILINE),
    # 'IDNO': re.compile(r'\b(?:N?1?I?D[\s#:_-]*N?O?|National\s*ID(?:\s*No)?)\b[:\.]?\s*([\d\s]{8,30})',re.IGNORECASE | re.MULTILINE)
'IDNO': re.compile(r'\b(?:(?:[IN1l|]{1,2}D|NID)(?:\s*NO|\s*No|\s*N0|[-_]?NO|NO|No|no|\s*N|\s*N0|[-_]?N|[-_]?N0|\s*#)?|NID\s*(?:Number|Num|NUM)|\|N0|\|NO|\|N)\b[:\.]?\s*([\d\s-]{8,30})', re.IGNORECASE | re.MULTILINE)

}

# Code 1 Functions (unchanged except for regex compilation)
def clean_header_text(text):
    keywords_to_remove = [
        "বাংলাদেশ সরকার", "জাতীয় পরিচয়", "জাতীয় পরিচয়", "National ID",
        "Government of the People's Republic", "জাতীয় পরিচয্ন পত্র"
    ]
    cleaned_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()
        if not any(keyword in line_stripped for keyword in keywords_to_remove):
            cleaned_lines.append(line_stripped)
    return "\n".join(cleaned_lines)

def infer_name_from_lines(text, extracted_fields):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if "Name" in line:
            if extracted_fields.get('নাম') in [None, "", "Not found"] and i > 0:
                extracted_fields['নাম'] = lines[i - 1]
            if extracted_fields.get('Name') in [None, "", "Not found"] and i + 1 < len(lines):
                extracted_fields['Name'] = lines[i + 1]
    return extracted_fields

def extract_fields_code1(text):
    extracted = {}
    for key, pattern in fields_code1.items():
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if not value:
                extracted[key] = "Not found"
                continue
            if key == 'IDNO':
                value = value.replace(" ", "")
            if len(value.split()) == 1 and len(value) < 3:
                extracted[key] = "Not found"
            else:
                extracted[key] = value
        else:
            extracted[key] = "Not found"
    return extracted

# Code 2 Functions (unchanged except for preprocessing)
class ImgCorrect:
    def __init__(self, img):
        self.img = img
        self.h, self.w, self.channel = self.img.shape
        if self.w <= self.h:
            self.scale = 700 / self.w
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.scale = 700 / self.h
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def img_lines(self):
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel)
        edges = cv2.Canny(binary, 50, 200)
        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        if self.lines is None:
            return None
        lines1 = self.lines[:, 0, :]
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return imglines

    def search_lines(self):
        lines = self.lines[:, 0, :]
        number_inexist_k = 0
        sum_pos_k45 = number_pos_k45 = 0
        sum_pos_k90 = number_pos_k90 = 0
        sum_neg_k45 = number_neg_k45 = 0
        sum_neg_k90 = number_neg_k90 = 0
        sum_zero_k = number_zero_k = 0

        for x in lines:
            if x[2] == x[0]:
                number_inexist_k += 1
                continue
            degree = degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if 0 < degree < 45:
                number_pos_k45 += 1
                sum_pos_k45 += degree
            if 45 <= degree < 90:
                number_pos_k90 += 1
                sum_pos_k90 += degree
            if -45 < degree < 0:
                number_neg_k45 += 1
                sum_neg_k45 += degree
            if -90 < degree <= -45:
                number_neg_k90 += 1
                sum_neg_k90 += degree
            if x[3] == x[1]:
                number_zero_k += 1

        max_number = max(number_inexist_k, number_pos_k45, number_pos_k90, number_neg_k45, number_neg_k90, number_zero_k)

        if max_number == number_inexist_k:
            return 90
        if max_number == number_pos_k45:
            return sum_pos_k45 / number_pos_k45
        if max_number == number_pos_k90:
            return sum_pos_k90 / number_pos_k90
        if max_number == number_neg_k45:
            return sum_neg_k45 / number_neg_k45
        if max_number == number_neg_k90:
            return sum_neg_k90 / number_neg_k90
        if max_number == number_zero_k:
            return 0

    def rotate_image(self, degree):
        if -45 <= degree <= 0:
            degree = degree
        if -90 <= degree < -45:
            degree = 90 + degree
        if 0 < degree <= 45:
            degree = degree
        if 45 < degree <= 90:
            degree = degree - 90
        height, width = self.img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        return imgRotation

def dskew(img):
    bg_color = [255, 255, 255]
    pad_img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=bg_color)
    imgcorrect = ImgCorrect(pad_img)
    lines_img = imgcorrect.img_lines()
    if lines_img is None:
        rotate = imgcorrect.rotate_image(0)
    else:
        degree = imgcorrect.search_lines()
        rotate = imgcorrect.rotate_image(degree)
    return rotate

def preprocess_before_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # sharpened = cv2.filter2D(gray, -1, kernel)
    # bilateral_filtered = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # contrast = clahe.apply(bilateral_filtered)
    # blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
    # adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return gray, img

def get_tesseract_ocr(image):#
    return ''

# def get_paddle_ocr(image):
#     results = ocr.ocr(image, cls=True)
#     all_text = ""
#     for line in results[0]:
#         text = line[1][0]
#         all_text += text + "\n"
#     return all_text

def get_easyocr_text(image_input):
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



def contains_english(text):
    if not text or text == "Not found":
        return False
    return bool(re.search(r'[a-zA-Z]', text))

def contains_bangla(text):
    if not text or text == "Not found":
        return False
    return bool(re.search(r'[\u0980-\u09FF]', text))

def clean_ocr_text(text):
    keywords_to_remove = [
        r"গণপ্রজাতন্ত্রী বাংলাদেশ সরকার", r"গণপ্রজাতন্ত্রী সরকার", r"গণপ্রজাতন্ত্রী",
        r"বাংলাদেশ সরকার", r"Government of the People", r"National ID Card",
        r"জাতীয় পরিচয় পত্র", r"জাতীয় পরিচয়", r"20/05/2025 09:12", r"জাতীয় পরিচয্ন পত্র"
    ]
    dob_pattern = re.compile(r"(Date of Birth|DOB|Date|Birth)[:：]?\s*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})", re.IGNORECASE)
    id_no_pattern = re.compile(r'\b(?:(?:[IN1l|]{1,2}D|NID)(?:\s*NO|\s*No|\s*N0|[-_]?NO|NO|No|no|\s*N|\s*N0|[-_]?N|[-_]?N0|\s*#)?|NID\s*(?:Num|NUM)|\|N0|\|NO|\|N)\b[:\.]?\s*([\d\s-]{8,30})', re.IGNORECASE | re.MULTILINE)
    dob_matches, id_no_matches = [], []

    def store_dob(match):
        dob_matches.append(match.group(0))
        return f"__DOB_{len(dob_matches) - 1}__"

    def store_id_no(match):
        id_no_matches.append(match.group(0))
        return f"__ID_NO_{len(id_no_matches) - 1}__"

    text = dob_pattern.sub(store_dob, text)
    text = id_no_pattern.sub(store_id_no, text)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if not line.strip():
            continue
        for keyword in keywords_to_remove:
            line = re.sub(keyword, "", line, flags=re.IGNORECASE)
        line = re.sub(r"[\[\]\(\)\{\}]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    for i, dob in enumerate(dob_matches):
        text = text.replace(f"__DOB_{i}__", dob)
    for i, id_no in enumerate(id_no_matches):
        text = text.replace(f"__ID_NO_{i}__", id_no)
    return text

def merge_lines(text):
    lines = text.splitlines()
    merged_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i].strip()
        if re.match(r"^[^\x00-\x7F]+$", current_line) and len(current_line) < 10 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if (re.match(r"^[^\x00-\x7F]+$", next_line) and
                    not any(pattern.search(next_line) for pattern in fields_code2.values())):
                merged_lines.append(current_line + " " + next_line)
                i += 2
                continue
        if re.match(r"^\d{1,2}$", current_line) and i + 1 < len(lines) and re.match(r"^\d{4}$", lines[i + 1]):
            merged_lines.append(current_line + " Jan " + lines[i + 1])
            i += 2
            continue
        merged_lines.append(current_line)
        i += 1
    return "\n".join(merged_lines)

def clean_bangla_name(name):
    if not name or name == "Not found":
        return "Not found"
    cleaned = re.sub(r"[^\u0980-\u09FF\s]", "", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned.split()) == 1 and len(cleaned) < 3:
        return "Not found"
    return cleaned if cleaned else "Not found"

def clean_english_name(name):
    if not name or name == "Not found":
        return "Not found"
    cleaned = re.sub(r"[^A-Za-z\s\-\.]", "", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned.split()) == 1 and len(cleaned) < 3:
        return "Not found"
    return cleaned if cleaned else "Not found"

def clean_date_of_birth(date):
    if not date or date == "Not found":
        return "Not found"
    cleaned = re.sub(r"[^0-9A-Za-z\s\-/]", "", date).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if re.match(
            r"^\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$",
            cleaned, re.IGNORECASE):
        year = int(re.search(r"\d{4}", cleaned).group())
        if 1900 <= year <= 2025:
            return cleaned
    return "Invalid"

def clean_id_no(id_no):
    if not id_no or id_no == "Not found":
        return "Not found"
    cleaned = re.sub(r"[^0-9]", "", id_no).strip()
    if re.match(r"^\d{10}$|^\d{13}$|^\d{17}$", cleaned):
        return cleaned
    return "Invalid"

def extract_fields_code2(text):
    extracted = {key: "Not found" for key in fields_code2}
    text = clean_ocr_text(text)
    text = merge_lines(text)

    for key, pattern in fields_code2.items():
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if not value:
                extracted[key] = "Not found"
                continue
            if len(value.split()) == 1 and len(value) < 3:
                extracted[key] = "Not found"
            else:
                extracted[key] = value

    lines = text.splitlines()
    name_index = -1
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if len(line.split()) == 1 and len(line) < 3:
            continue
        if not any(pattern.search(line) for pattern in fields_code2.values()):
            if re.match(r"[^\x00-\x7F]+", line) and extracted["নাম"] == "Not found":
                extracted["নাম"] = line
                name_index = i
            elif re.match(r"[A-Za-z\s\.]+", line) and extracted["Name"] == "Not found" and (
                    name_index == -1 or i > name_index):
                extracted["Name"] = line
            elif re.match(r"[^\x00-\x7F]+", line) and extracted["পিতা"] == "Not found":
                if (extracted["Name"] != "Not found" and i > lines.index(extracted["Name"]) if extracted["Name"] in lines else True) or \
                        (extracted["নাম"] != "Not found" and i > lines.index(extracted["নাম"]) if extracted["নাম"] in lines else i > name_index):
                    extracted["পিতা"] = line
            elif re.match(r"[^\x00-\x7F]+", line) and extracted["মাতা"] == "Not found" and extracted["পিতা"] != "Not found":
                if (extracted["পিতা"] != "Not found" and i > lines.index(extracted["পিতা"]) if extracted["পিতা"] in lines else True) or \
                        (extracted["Name"] != "Not found" and i > lines.index(extracted["Name"]) if extracted["Name"] in lines else True) or \
                        (extracted["নাম"] != "Not found" and i > lines.index(extracted["নাম"]) if extracted["নাম"] in lines else i > name_index):
                    extracted["মাতা"] = line

    extracted["নাম"] = clean_bangla_name(extracted["নাম"])
    extracted["পিতা"] = clean_bangla_name(extracted["পিতা"])
    extracted["মাতা"] = clean_bangla_name(extracted["মাতা"])
    extracted["স্বামী"] = clean_bangla_name(extracted["স্বামী"])
    extracted["স্ত্রী"] = clean_bangla_name(extracted["স্ত্রী"])
    extracted["Name"] = clean_english_name(extracted["Name"])
    extracted["DateOfBirth"] = clean_date_of_birth(extracted["DateOfBirth"])
    extracted["IDNO"] = clean_id_no(extracted["IDNO"])

    fields_to_validate = ['নাম', 'পিতা', 'মাতা', 'স্বামী', 'স্ত্রী']
    for field in fields_to_validate:
        if contains_english(extracted[field]):
            extracted[field] = "Not found"
    if contains_bangla(extracted["Name"]):
        extracted["Name"] = "Not found"

    return extracted

# def remove_special_chars(text, field):
#     if text == "Not found" or not text:
#         return text
#     if field == "DateOfBirth":
#         cleaned = re.sub(r"[^0-9A-Za-z\s\-/\.]", "", text).strip()
#         cleaned = re.sub(r"[-/]", " ", cleaned).strip()
#         return re.sub(r"\s+", " ", cleaned).strip()
#     if field == "IDNO":
#         return re.sub(r"[^0-9]", "", text).strip()
#     cleaned = re.sub(r"[^A-Za-z\s\.\u0980-\u09FF]", "", text).strip()
#     return re.sub(r"\s+", " ", cleaned).strip()
def remove_special_chars(text, field):
    if text == "Not found" or not text:
        return text
    if field == "DateOfBirth":
        cleaned = re.sub(r"[^0-9A-Za-z\s\-]", "", text).strip()
        cleaned = re.sub(r"[-]", " ", cleaned).strip()
        return re.sub(r"\s+", " ", cleaned).strip()
    if field == "IDNO":
        return re.sub(r"[^0-9]", "", text).strip()
    cleaned = re.sub(r"[^A-Za-z\s\.\-\u0980-\u09FF]", "", text).strip()
    return re.sub(r"\s+", " ", cleaned).strip()

def clean_date_field(text):
    #print(text)
    if text == "Not found" or not text:
        return text
    month_map = {
        'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Apr',
        'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug',
        'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dec',
        'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr',
        'may': 'May', 'june': 'Jun', 'july': 'Jul', 'august': 'Aug',
        'september': 'Sep', 'october': 'Oct', 'november': 'Nov', 'december': 'Dec'
    }
    date_pattern = r'(\d{1,2})\s*[-/\s]?([A-Za-z]+)\s*[-/\s]?(\d{4})'
    match = re.match(date_pattern, text.strip(), re.IGNORECASE)
    if match:
        day, month, year = match.groups()
        month = month[:3].lower()
        if month in month_map:
            month = month_map[month]
        else:
            return "Invalid"
        current_year = datetime.now().year
        if not (1800 <= int(year) <= current_year - 18):
            return "Invalid"
        day = int(day)
        if not (1 <= day <= 31):
            return "Invalid"
        return f"{day:02d} {month} {year}"
    return "Invalid"

# def clean_all_special_chars(text):
#     if text == "Not found" or not text:
#         return text
#     cleaned = re.sub(r"[^A-Za-z0-9\s\.\u0980-\u09FF]", "", text).strip()
#     return re.sub(r"\s+", " ", cleaned).strip()
def clean_all_special_chars(text):
    if text == "Not found" or not text:
        return text
    cleaned = re.sub(r"[^A-Za-z0-9\s\.\-\u0980-\u09FF]", "", text).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_bangla_field(text):
    if text == "Not found" or not text:
        return text
    words = text.split()
    cleaned_words = [word for word in words if not re.search(r'[A-Za-z]', word)]
    cleaned = " ".join(cleaned_words).strip()
    return cleaned if cleaned else "Not found"

def clean_english_field(text):
    if text == "Not found" or not text:
        return text
    words = text.split()
    cleaned_words = [word for word in words if not re.search(r'[\u0980-\u09FF]', word)]
    cleaned = " ".join(cleaned_words).strip()
    cleaned = re.sub(r'\.(\w)', r'. \1', cleaned)
    return cleaned if cleaned else "Not found"



def compare_outputs(t1, t2, t3, field): # Removed p1/comapsimn from parameters
    t1 = "Not found" if t1 == "No data found" else t1
    t2 = "Not found" if t2 == "No data found" else t2
    t3 = "Not found" if t3 == "No data found" else t3

    outputs_raw = [t1, t2, t3] # Only t1, t2, t3
    outputs = []
    for val in outputs_raw:
        if not val or val == "":
            outputs.append("Not found")
        else:
            outputs.append(val)

    if field == "DateOfBirth":
        outputs = [clean_date_field(val) for val in outputs]
        outputs_raw = [clean_date_field(val) for val in outputs_raw]

        # For DateOfBirth, handle Invalid explicitly
        valid_dates = [val for val in outputs if val != "Invalid" and val != "Not found"]
        if not valid_dates:  # All outputs are Invalid or Not found
            return "Invalid"
        if len(valid_dates) == 1:  # Only one valid date
            return valid_dates[0]
        # If multiple valid dates, proceed with comparison logic below
        outputs = [val if val not in ["Invalid", "Not found"] else "Not found" for val in outputs]
        outputs_raw = [val if val not in ["Invalid", "Not found"] else "Not found" for val in outputs_raw]

    outputs_cleaned = [clean_all_special_chars(val) for val in outputs]
    outputs_raw = [clean_all_special_chars(val) for val in outputs_raw]

    if field in ['নাম', 'পিতা', 'মাতা', 'স্বামী', 'স্ত্রী']:
        outputs_cleaned = [clean_bangla_field(val) for val in outputs_cleaned]
        outputs_raw = [clean_bangla_field(val) for val in outputs_raw]
    elif field == "Name":
        outputs_cleaned = [clean_english_field(val) for val in outputs_cleaned]
        outputs_raw = [clean_english_field(val) for val in outputs_raw]

    outputs_cleaned = [remove_special_chars(val, field) for val in outputs_cleaned]

    outputs_final = []
    for val in outputs_cleaned:
        if val != "Not found" and len(val.split()) == 1 and len(val) < 3:
            outputs_final.append("Not found")
        else:
            outputs_final.append(val)

    # print("\n================= OCR COMPARISON =================")
    # # Adjusted print statement for 3 inputs
    # print(f"{'':<17} t1                        | t2                        | t3")
    # print(f"{'Raw Outputs':<17}: {outputs_raw[0]:<25} | {outputs_raw[1]:<25} | {outputs_raw[2]}")
    # print(
    #     f"{'Cleaned Outputs':<17}: {outputs_final[0]:<25} | {outputs_final[1]:<25} | {outputs_final[2]}")
    # print("==================================================\n")

    if all(val == "Not found" for val in outputs_final):
        return "Not found"

    valid_outputs = [val for val in outputs_final if val != "Not found"]
    valid_raw = [val for val in outputs_raw if val != "Not found"]

    if len(valid_outputs) == 1:
        return valid_outputs[0]

    pairs = [
        (outputs_final[0], outputs_final[1], outputs_raw[0], outputs_raw[1]), # t1 vs t2
        (outputs_final[0], outputs_final[2], outputs_raw[0], outputs_raw[2]), # t1 vs t3
        (outputs_final[1], outputs_final[2], outputs_raw[1], outputs_raw[2])  # t2 vs t3
    ]
    matching_pairs = [(v1, v2, r1, r2) for v1, v2, r1, r2 in pairs if v1 == v2 and v1 != "Not found"]

    if len(matching_pairs) >= 1:
        best_value = None
        max_words = -1 # Initialize with -1 for words count to ensure any valid word count is higher
        has_special = True # Assume special chars initially to prioritize no special chars

        for v1, _, r1, r2 in matching_pairs: # v2 and r2 are used for checking special chars in the paired raw output
            words1 = len(v1.split())
            special1_in_raw1 = bool(re.search(r"[^A-Za-z\s\.\u0980-\u09FF0-9]", r1))
            special2_in_raw2 = bool(re.search(r"[^A-Za-z\s\.\u0980-\u09FF0-9]", r2))

            # Determine if the current matching pair is "better"
            if best_value is None: # If it's the first matching pair found
                best_value = v1
                max_words = words1
                has_special = special1_in_raw1 or special2_in_raw2
            else:
                # Prioritize values without special characters
                current_has_special = special1_in_raw1 or special2_in_raw2
                if not current_has_special and has_special: # Current has no special, previous had special
                    best_value = v1
                    max_words = words1
                    has_special = False
                elif current_has_special == has_special: # Both have or don't have special chars
                    if words1 > max_words: # Pick the one with more words
                        best_value = v1
                        max_words = words1
                        has_special = current_has_special # Update special char status
        if best_value:
            return best_value

    value_counts = {}
    for val in valid_outputs:
        value_counts[val] = value_counts.get(val, 0) + 1
    matching_values = [val for val, count in value_counts.items() if count >= 2]
    if matching_values:
        # For 3 inputs, if count >= 2, there will be only one such value.
        return matching_values[0]

    not_found_count = outputs_final.count("Not found")
    if not_found_count in [1, 2] and len(valid_outputs) in [1, 2]: # Adjusted for 3 inputs
        if len(valid_outputs) == 1:
            return valid_outputs[0]
        if len(valid_outputs) == 2:
            words1 = len(valid_outputs[0].split())
            words2 = len(valid_outputs[1].split())
            return valid_outputs[0] if words1 >= words2 else valid_outputs[1]

    unique_outputs = list(set(valid_outputs))
    if len(unique_outputs) == len(valid_outputs):
        for val in unique_outputs:
            word_count = len(val.split())
            if word_count in [3, 4]:
                return val

    max_words = 0
    best_val = valid_outputs[0] if valid_outputs else "Not found"
    for val in valid_outputs:
        words = len(val.split())
        if words > max_words:
            max_words = words
            best_val = val
    return best_val

def process_image(image_path):
    # code 1

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise ValueError(f"Failed to load image at {image_path}")

    easyocr_text1 = get_easyocr_text(image_path)
    #print(easyocr_text1)
    raw1= easyocr_text1
    # easyocr_text1 = clean_header_text(easyocr_text1)
    # easyocr_results1 = extract_fields_code1(easyocr_text1)
    # easyocr_results1 = infer_name_from_lines(easyocr_text1, easyocr_results1)

    # paddle_text1 = get_paddle_ocr(img_cv2)
    # print("PaddleOCR Code 1:", paddle_text1)
    # paddle_text1_cleaned = clean_header_text(paddle_text1)
    # paddle_results1 = extract_fields_code2(paddle_text1_cleaned)
    # paddle_results1 = infer_name_from_lines(paddle_text1_cleaned, paddle_results1)

    # Preprocess (Deskew) for subsequent runs
    rotated_img = dskew(img_cv2)

    # Code 2 Processing with Preprocessing
    preprocessed_img, _ = preprocess_before_crop(rotated_img)
    easyocr_text2 = get_easyocr_text(preprocessed_img)
    #print("EasyOCR Deskewed Text:", easyocr_text2)
    raw2=easyocr_text2
    easyocr_results2 = extract_fields_code2(easyocr_text2)


    # Code 3 Processing with Preprocessing (just rotate)
    easyocr_text3 = get_easyocr_text(rotated_img)
    #print(easyocr_text3)
    raw3=easyocr_text3
    # easyocr_text3 = clean_header_text(easyocr_text3)
    # easyocr_results3 = extract_fields_code1(easyocr_text3)
    # easyocr_results3 = infer_name_from_lines(easyocr_text3, easyocr_results3)

    all_raw_easyocr_data = (
        f"RAW DOC 1: {raw1} | "
        f"RAW DOC 2: {raw2} | "
        f"RAW DOC 3: {raw3}"
    )
    # Combine results
    # final_results = {}
    # for field in fields_code1:
    #     e1 = easyocr_results1.get(field, "Not found")
    #     # p1 = paddle_results1.get(field, "Not found")
    #     e2 = easyocr_results2.get(field, "Not found")
    #     e3 = easyocr_results3.get(field, "Not found")
    #     final_results[field] = compare_outputs(e1,e2,e3,field)

    # for field, value in final_results.items():
    #     print(f"{field}: {value}")

    return all_raw_easyocr_data

# Example Usage
# image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_15.png"
# final_results = process_image(image_path)

# Example Usage
image_path = sys.argv[1]
final_results1 = process_image(image_path)
print(json.dumps(final_results1, ensure_ascii=False))