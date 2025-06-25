import cv2
import numpy as np
import pytesseract
import easyocr
import re
from PIL import Image
from math import atan, degrees, radians, sin, cos, fabs
import warnings
from paddleocr import PaddleOCR

# Global OCR engine initialization
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
reader = easyocr.Reader(['en', 'bn'], gpu=False, model_storage_directory=r'C:\easy\model',
                        user_network_directory=r'C:\easy\network')
warnings.filterwarnings("ignore", category=UserWarning, module="paddle.utils.cpp_extension")
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

# Unified Regex Patterns
fields = {
    'নাম': r'নাম[:：]*\s*([^\n:]+)',
    'Name': r'Name[:：]*\s*([^\n:]+)',
    'পিতা': r'পিতা[:：]*\s*([^\n:]+)',
    'মাতা': r'মাতা[:：]*\s*([^\n:]+)',
    'স্বামী': r'(?:স্বামী|স্বা[:;মী-]*|husband|sami)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্ত্রী|Date|ID)',
    'স্ত্রী': r'(?:স্ত্রী|স্ত্র[:;ী-]*|wife|stri)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|Date|ID)',
    'DateOfBirth': r'(?:Date of Birth|DOB|Date|Birth)[:;\s-]*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|স্ত্রী|ID)',
    'IDNO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]*\s*([\d\s-]{8,30})'
}


# Image Correction Class
class ImgCorrect:
    def __init__(self, img):
        self.img = img
        self.h, self.w = img.shape[:2]
        scale = 700 / min(self.w, self.h)
        self.img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect_lines(self):
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel)
        edges = cv2.Canny(binary, 50, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        return lines[:, 0, :] if lines is not None else None

    def calculate_rotation_angle(self, lines):
        if lines is None:
            return 0
        angles = {'inexist': 0, 'pos_k45': 0, 'pos_k90': 0, 'neg_k45': 0, 'neg_k90': 0, 'zero': 0}
        counts = {key: 0 for key in angles}
        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                counts['inexist'] += 1
                continue
            if y2 == y1:
                counts['zero'] += 1
                continue
            degree = degrees(atan((y2 - y1) / (x2 - x1)))
            if 0 < degree < 45:
                counts['pos_k45'] += 1
                angles['pos_k45'] += degree
            elif 45 <= degree < 90:
                counts['pos_k90'] += 1
                angles['pos_k90'] += degree
            elif -45 < degree < 0:
                counts['neg_k45'] += 1
                angles['neg_k45'] += degree
            elif -90 < degree <= -45:
                counts['neg_k90'] += 1
                angles['neg_k90'] += degree
        max_count_key = max(counts, key=counts.get)
        if max_count_key == 'inexist':
            return 90
        elif max_count_key == 'zero':
            return 0
        return angles[max_count_key] / counts[max_count_key] if counts[max_count_key] else 0

    def rotate_image(self, degree):
        if degree == 0:
            return self.img
        if degree == 90:
            return cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        height, width = self.img.shape[:2]
        height_new = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        width_new = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        mat_rotation[0, 2] += (width_new - width) / 2
        mat_rotation[1, 2] += (height_new - height) / 2
        return cv2.warpAffine(self.img, mat_rotation, (width_new, height_new), borderValue=(255, 255, 255))


# Preprocessing Functions
def preprocess_image(img, full_preprocess=False):
    img_correct = ImgCorrect(img)
    lines = img_correct.detect_lines()
    rotated_img = img_correct.rotate_image(img_correct.calculate_rotation_angle(lines))

    if full_preprocess:
        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        bilateral_filtered = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(bilateral_filtered)
        blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return rotated_img, adaptive_thresh
    return rotated_img, None


def clean_ocr_text(text):
    if not text:
        return ""
    keywords_to_remove = [
        r"গণপ্রজাতন্ত্রী বাংলাদেশ সরকার", r"গণপ্রজাতন্ত্রী সরকার", r"গণপ্রজাতন্ত্রী",
        r"বাংলাদেশ সরকার", r"Government of the People", r"National ID Card",
        r"জাতীয় পরিচয় পত্র", r"জাতীয় পরিচয়", r"20/05/2025 09:12", r"জাতীয় পরিচয্ন পত্র"
    ]
    dob_pattern = r"(Date of Birth|DOB|Date|Birth)[:：]?\s*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    id_no_pattern = r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]*\s*([\d\s-]{8,30})'

    dob_matches, id_no_matches = [], []

    def store_dob(match):
        dob_matches.append(match.group(0))
        return f"__DOB_{len(dob_matches) - 1}__"

    def store_id_no(match):
        id_no_matches.append(match.group(0))
        return f"__ID_NO_{len(id_no_matches) - 1}__"

    text = re.sub(dob_pattern, store_dob, text, flags=re.IGNORECASE)
    text = re.sub(id_no_pattern, store_id_no, text, flags=re.IGNORECASE)
    lines = [re.sub(r"|".join(keywords_to_remove), "", line, flags=re.IGNORECASE).strip() for line in text.splitlines()
             if line.strip()]
    text = "\n".join(line for line in lines if not re.match(r"[\[\]\(\)\{\}]", line))
    for i, dob in enumerate(dob_matches):
        text = text.replace(f"__DOB_{i}__", dob)
    for i, id_no in enumerate(id_no_matches):
        text = text.replace(f"__ID_NO_{i}__", id_no)
    return text


def merge_lines(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    merged_lines, i = [], 0
    while i < len(lines):
        current_line = lines[i]
        if re.match(r"^[^\x00-\x7F]+$", current_line) and len(current_line) < 10 and i + 1 < len(lines):
            next_line = lines[i + 1]
            if re.match(r"^[^\x00-\x7F]+$", next_line) and not any(
                    re.search(pattern, next_line, re.IGNORECASE) for pattern in fields.values()):
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


def clean_field(value, field):
    if not value or value == "Not found":
        return "Not found"
    if field == "IDNO":
        cleaned = re.sub(r"[^0-9]", "", value).strip()
        return cleaned if re.match(r"^\d{10}$|^\d{13}$|^\d{17}$", cleaned) else "Invalid"
    if field == "DateOfBirth":
        cleaned = re.sub(r"[^0-9A-Za-z\s\-/]", "", value).strip()
        if re.match(
                r"^\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$",
                cleaned, re.IGNORECASE):
            year = int(re.search(r"\d{4}", cleaned).group())
            return cleaned if 1900 <= year <= 2025 else "Invalid"
        return "Invalid"
    if field in ['নাম', 'পিতা', 'মাতা', 'স্বামী', 'স্ত্রী']:
        cleaned = re.sub(r"[^\u0980-\u09FF\s]", "", value).strip()
        if contains_english(cleaned):
            return "Not found"
    elif field == "Name":
        cleaned = re.sub(r"[^A-Za-z\s\.]", "", value).strip()
        if contains_bangla(cleaned):
            return "Not found"
    else:
        cleaned = re.sub(r"[^A-Za-z\s\.\u0980-\u09FF]", "", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return "Not found" if len(cleaned.split()) == 1 and len(cleaned) < 3 else cleaned


def contains_english(text):
    return bool(re.search(r'[a-zA-Z]', text)) if text and text != "Not found" else False


def contains_bangla(text):
    return bool(re.search(r'[\u0980-\u09FF]', text)) if text and text != "Not found" else False


def extract_fields(text, use_advanced=True):
    text = clean_ocr_text(text)
    if use_advanced:
        text = merge_lines(text)
    extracted = {key: "Not found" for key in fields}

    # Regex-based extraction
    for key, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value and not (len(value.split()) == 1 and len(value) < 3):
                extracted[key] = value

    # Line-based inference for remaining fields
    if use_advanced:
        lines = text.splitlines()
        name_index = -1
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or (len(line.split()) == 1 and len(line) < 3):
                continue
            if not any(re.search(pattern, line, re.IGNORECASE) for pattern in fields.values()):
                if re.match(r"[^\x00-\x7F]+", line) and extracted["নাম"] == "Not found":
                    extracted["নাম"] = line
                    name_index = i
                elif re.match(r"[A-Za-z\s\.]+", line) and extracted["Name"] == "Not found" and (
                        name_index == -1 or i > name_index):
                    extracted["Name"] = line
                elif re.match(r"[^\x00-\x7F]+", line) and extracted["পিতা"] == "Not found":
                    if (extracted["নাম"] != "Not found" and i > lines.index(extracted["নাম"]) if extracted[
                                                                                                     "নাম"] in lines else i > name_index):
                        extracted["পিতা"] = line
                elif re.match(r"[^\x00-\x7F]+", line) and extracted["মাতা"] == "Not found" and extracted[
                    "পিতা"] != "Not found":
                    if (extracted["পিতা"] != "Not found" and i > lines.index(extracted["পিতা"]) if extracted[
                                                                                                       "পিতা"] in lines else True):
                        extracted["মাতা"] = line

    # Clean extracted fields
    return {key: clean_field(value, key) for key, value in extracted.items()}


def compare_outputs(outputs, field):
    cleaned_outputs = [clean_field(val, field) for val in outputs]
    valid_outputs = [val for val in cleaned_outputs if val not in ["Not found", "Invalid"]]

    if not valid_outputs:
        return "Not found"
    if len(valid_outputs) == 1:
        return valid_outputs[0]

    value_counts = {}
    for val in valid_outputs:
        value_counts[val] = value_counts.get(val, 0) + 1
    matching_values = [val for val, count in value_counts.items() if count >= 2]
    if matching_values:
        return matching_values[0]

    return max(valid_outputs, key=lambda x: len(x.split()))


def get_ocr_texts(image_path, rotated_img, preprocessed_img, rotated_img_pil):
    # Original image OCR
    img_pil = Image.open(image_path)
    tesseract_text1 = pytesseract.image_to_string(img_pil, lang='ben+eng')
    paddle_text1 = "\n".join(line[1][0] for line in ocr.ocr(image_path, cls=True)[0]) if ocr.ocr(image_path,
                                                                                                 cls=True) else ""
    easyocr_text1 = "\n".join(text for _, text, _ in reader.readtext(image_path))

    # Rotated image OCR
    tesseract_text2 = pytesseract.image_to_string(rotated_img_pil, lang='ben+eng')
    easyocr_text2 = "\n".join(text for _, text, _ in reader.readtext(rotated_img))

    # Preprocessed image OCR
    tesseract_text3 = pytesseract.image_to_string(Image.fromarray(preprocessed_img),
                                                  lang='ben+eng') if preprocessed_img is not None else ""
    easyocr_text3 = "\n".join(
        text for _, text, _ in reader.readtext(preprocessed_img)) if preprocessed_img is not None else ""

    return [tesseract_text1, paddle_text1, easyocr_text1, tesseract_text2, easyocr_text2, tesseract_text3,
            easyocr_text3]


def process_image(image_path):
    # Load and preprocess image
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        return {key: "Not found" for key in fields}

    # Perform preprocessing
    rotated_img, preprocessed_img = preprocess_image(img_cv2, full_preprocess=True)
    rotated_img_pil = Image.fromarray(rotated_img)

    # Get OCR texts
    ocr_texts = get_ocr_texts(image_path, rotated_img, preprocessed_img, rotated_img_pil)

    # Extract fields
    results = [
        extract_fields(ocr_texts[0], use_advanced=False),  # Tesseract original
        extract_fields(ocr_texts[1], use_advanced=False),  # PaddleOCR original
        extract_fields(ocr_texts[2], use_advanced=False),  # EasyOCR original
        extract_fields(ocr_texts[3], use_advanced=True),  # Tesseract rotated
        extract_fields(ocr_texts[4], use_advanced=True),  # EasyOCR rotated
        extract_fields(ocr_texts[5], use_advanced=True),  # Tesseract preprocessed
        extract_fields(ocr_texts[6], use_advanced=True)  # EasyOCR preprocessed
    ]

    # Combine results
    final_results = {}
    for field in fields:
        outputs = [r.get(field, "Not found") for r in results]
        final_results[field] = compare_outputs(outputs, field)

    return final_results


# Example Usage
if __name__ == "__main__":
    image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_2.png"
    final_results = process_image(image_path)
    print(final_results)