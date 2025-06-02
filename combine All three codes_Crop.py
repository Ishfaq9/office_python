import cv2
import numpy as np
import os
import pytesseract
import easyocr
import re
import json
import sys
from PIL import Image
from math import atan, degrees, radians, sin, cos, fabs
from skimage import color, morphology, measure

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
reader = easyocr.Reader(['en', 'bn'], gpu=False)

# Code 1 Regex Patterns
fields_code1 = {
    'নাম': r'নাম[:：]*\s*([^\n:]+)',
    'Name': r'Name[:：]*\s*([^\n:]+)',
    'পিতা': r'পিতা[:：]*\s*([^\n:]+)',
    'মাতা': r'মাতা[:：]*\s*([^\n:]+)',
    'স্বামী': r'স্বামী[:：]*\s*([^\n:]+)',
    'স্ত্রী': r'স্ত্রী[:：]*\s*([^\n:]+)',
    'DateOfBirth': r'Date of Birth[:：]*\s*([^\n:]+)',
    'IDNO': r'(?:ID\s*NO|NID\s*No\.?|ID|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]*\s*([\d\s]{8,30})'
}

# Code 2 Regex Patterns
fields_code2 = {
    'নাম': r'নাম[:：]*\s*([^\n:]+)',
    'Name': r'Name[:：]*\s*([^\n:]+)',
    'পিতা': r'পিতা[:：]*\s*([^\n:]+)',
    'মাতা': r'মাতা[:：]*\s*([^\n:]+)',
    'স্বামী': r'(?:স্বামী|স্বা[:;মী-]*|husband|sami)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্ত্রী|Date|ID)',
    'স্ত্রী': r'(?:স্ত্রী|স্ত্র[:;ী-]*|wife|stri)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|Date|ID)',
    'DateOfBirth': r'(?:Date of Birth|DOB|Date|Birth)[:;\s-]*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|স্ত্রী|ID)',
    'IDNO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]*\s*([\d\s]{8,30})'
}


# Code 1 Functions
def crop_image(image_bgr):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert RGB to HSV and segment on low saturation
    image_hsv = color.rgb2hsv(image_rgb)
    seg = image_hsv[:, :, 1] < 0.10

    # Morphological operations
    seg_cleaned = morphology.opening(seg, morphology.disk(1))
    seg_cleaned = morphology.closing(seg_cleaned, morphology.disk(25))

    # Extract largest connected component
    def get_main_component(segments):
        labels = measure.label(segments)
        if labels.max() == 0:
            return segments
        return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    background = get_main_component(~seg_cleaned)
    filled = ~background

    # Final cleaned mask
    mask = morphology.opening(filled, morphology.disk(25))

    # Apply mask to original image
    masked_result = image_rgb.copy()
    masked_result[~mask] = 0

    # Get bounding box and crop
    mask_x = mask.max(axis=0)
    mask_y = mask.max(axis=1)
    indices_x = mask_x.nonzero()[0]
    indices_y = mask_y.nonzero()[0]

    if len(indices_x) == 0 or len(indices_y) == 0:
        return image_rgb  # Return original if nothing found

    minx, maxx = int(indices_x[0]), int(indices_x[-1])
    miny, maxy = int(indices_y[0]), int(indices_y[-1])

    cropped_result = image_rgb[miny:maxy, minx:maxx, :]
    return cropped_result


def clean_header_text(text):
    keywords_to_remove = [
        "বাংলাদেশ সরকার", "জাতীয় পরিচয়", "জাতীয় পরিচয়", "National ID",
        "Government of the People's Republic"
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
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if key == 'IDNO':
                value = value.replace(" ", "")  # Remove spaces from NID
            extracted[key] = value
        else:
            extracted[key] = "Not found"
    return extracted


# Code 2 Functions
#------------ rotate part
class ImgCorrect():
    def __init__(self, img):
        self.img = img
        self.h, self.w, self.channel = self.img.shape
        # print("Original images h & w -> | w: ",self.w, "| h: ",self.h)
        if self.w <= self.h:
            self.scale = 700 / self.w
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.scale = 700 / self.h
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        #print("Resized Image by Padding and Scaling:")
        #plot_fig(self.img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def img_lines(self):
        #print("Gray Image:")
        #plot_fig(self.gray)
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("bin",binary)
        #print("Inverse Binary:")
        #plot_fig(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # rectangular structure
        # print("Kernel for dialation:")
        # print(kernel)
        binary = cv2.dilate(binary, kernel)  # dilate
        #print("Dilated Binary:")
        #plot_fig(binary)
        edges = cv2.Canny(binary, 50, 200)
        #print("Canny edged detection:")
        #plot_fig(edges)

        # print("Edge 1: ")
        # cv2.imshow("edges", edges)

        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        # print(self.lines)
        if self.lines is None:
            #print("Line segment not found")
            return None

        lines1 = self.lines[:, 0, :]  # Extract as 2D
        # print(lines1)
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #print("Probabilistic Hough Lines:")
        #plot_fig(imglines)
        return imglines

    def search_lines(self):
      lines = self.lines[:, 0, :]  # extract as 2D

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
          #print(degrees(atan((x[3] - x[1]) / (x[2] - x[0]))), "pos:", x[0], x[1], x[2], x[3], "Slope:",(x[3] - x[1]) / (x[2] - x[0]))
          degree = degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
          # print("Degree or Slope of detected lines : ",degree)
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

      max_number = max(number_inexist_k, number_pos_k45, number_pos_k90, number_neg_k45,number_neg_k90, number_zero_k)
      # print("Num of lines in different Degree range ->")
      # print("Not a Line: ",number_inexist_k, "| 0 to 45: ",number_pos_k45, "| 45 to 90: ",number_pos_k90, "| -45 to 0: ",number_neg_k45, "| -90 to -45: ",number_neg_k90, "| Line where y1 equals y2 :",number_zero_k)

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
        """
        Positive angle counterclockwise rotation
        :param degree:
        :return:
        """
        # print("degree:", degree)
        if -45 <= degree <= 0:
            degree = degree  # #negative angle clockwise
        if -90 <= degree < -45:
            degree = 90 + degree  # positive angle counterclockwise
        if 0 < degree <= 45:
            degree = degree  # positive angle counterclockwise
        if 45 < degree <= 90:
            degree = degree - 90  # negative angle clockwise
        #print("DSkew angle: ", degree)

        # degree = degree - 90
        height, width = self.img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(
            cos(radians(degree))))  # This formula refers to the previous content
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        # print("Height :",height)
        # print("Width :",width)
        # print("HeightNew :",heightNew)
        # print("WidthNew :",widthNew)

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # rotate degree counterclockwise
        # print("Mat Rotation (Before): ",matRotation)
        matRotation[0, 2] += (widthNew - width) / 2
        # Because after rotation, the origin of the coordinate system is the upper left corner of the new image, so it needs to be converted according to the original image
        matRotation[1, 2] += (heightNew - height) / 2
        # print("Mat Rotation (After): ",matRotation)

        # Affine transformation, the background color is filled with white
        imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        # Padding
        pad_image_rotate = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(0, 255, 0))
        #plot_fig(pad_image_rotate)

        return imgRotation



def dskew(img):
    #img_loc = line_path + img
    #im = cv2.imread(img_loc)
    im=img
    # Padding
    bg_color = [255, 255, 255]
    pad_img = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value=bg_color)

    imgcorrect = ImgCorrect(pad_img)
    lines_img = imgcorrect.img_lines()
    # print(type(lines_img))

    if lines_img is None:
        rotate = imgcorrect.rotate_image(0)
    else:
        degree = imgcorrect.search_lines()
        rotate = imgcorrect.rotate_image(degree)


    return rotate


# ✅ Preprocessing function (fixed version)
def preprocess_before_crop(scan_path):
    #original_image = cv2.imread(scan_path)
    original_image = scan_path

    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Skip equalizeHist to avoid background merging problems

    # Initial Denoising
    #denoised = cv2.fastNlMeansDenoising(gray, None, h=20, templateWindowSize=7, searchWindowSize=21)

    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # Bilateral filter
    bilateral_filtered = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

    # Contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(bilateral_filtered)

    # Blur
    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)

    # Adaptive Threshold
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    return gray, original_image




def get_tesseract_ocr(image):
    img = Image.fromarray(image)
    ocr_text = pytesseract.image_to_string(img, lang='ben+eng')
    return ocr_text


def get_easyocr_text(image_path):
    results = reader.readtext(image_path)
    ocr_text = "\n".join([text for _, text, _ in results])
    return ocr_text


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
        r"জাতীয় পরিচয় পত্র", r"জাতীয় পরিচয়", r"20/05/2025 09:12"
    ]
    dob_pattern = r"(Date of Birth|DOB|Date|Birth)[:：]?\s*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    id_no_pattern = r"(ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]?\s*([\d ]{8,30})"
    dob_matches, id_no_matches = [], []

    def store_dob(match):
        dob_matches.append(match.group(0))
        return f"__DOB_{len(dob_matches) - 1}__"

    def store_id_no(match):
        id_no_matches.append(match.group(0))
        return f"__ID_NO_{len(id_no_matches) - 1}__"

    text = re.sub(dob_pattern, store_dob, text, flags=re.IGNORECASE)
    text = re.sub(id_no_pattern, store_id_no, text, flags=re.IGNORECASE)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if not line.strip():
            continue
        for keyword in keywords_to_remove:
            line = re.sub(keyword, "", line, flags=re.IGNORECASE)
        line = re.sub(r"[\[\]\(\)\{\}0-9]{3,}", "", line)
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
                    not any(re.search(pattern, next_line, re.IGNORECASE) for pattern in fields_code2.values())):
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
        return name
    cleaned = re.sub(r"[^\u0980-\u09FF\s]", "", name).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_english_name(name):
    if not name or name == "Not found":
        return name
    cleaned = re.sub(r"[^A-Za-z\s\.]", "", name).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_date_of_birth(date):
    if not date or date == "Not found":
        return date
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
        return id_no
    cleaned = re.sub(r"[^0-9]", "", id_no).strip()
    if re.match(r"^\d{10}$|^\d{13}$|^\d{17}$", cleaned):
        return cleaned
    return "Invalid"


def extract_fields_code2(text):
    extracted = {key: "Not found" for key in fields_code2}
    text = clean_ocr_text(text)
    text = merge_lines(text)
    for key, pattern in fields_code2.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()
    lines = text.splitlines()
    name_index = -1
    for i, line in enumerate(lines):
        if not any(re.search(pattern, line, re.IGNORECASE) for pattern in fields_code2.values()):
            if re.match(r"[^\x00-\x7F]+", line) and extracted["নাম"] == "Not found":
                extracted["নাম"] = line.strip()
                name_index = i
            elif re.match(r"[A-Za-z\s\.]+", line) and extracted["Name"] == "Not found" and (
                    name_index == -1 or i > name_index):
                extracted["Name"] = line.strip()
            elif (re.match(r"[^\x00-\x7F]+", line) and extracted["পিতা"] == "Not found"):
                if (extracted["Name"] != "Not found" and i > lines.index(extracted["Name"]) if extracted[
                                                                                                   "Name"] in lines else True) or \
                        (extracted["নাম"] != "Not found" and i > lines.index(extracted["নাম"]) if extracted[
                                                                                                      "নাম"] in lines else i > name_index):
                    extracted["পিতা"] = line.strip()
            elif (re.match(r"[^\x00-\x7F]+", line) and extracted["মাতা"] == "Not found" and extracted[
                "পিতা"] != "Not found"):
                if (extracted["পিতা"] != "Not found" and i > lines.index(extracted["পিতা"]) if extracted[
                                                                                                   "পিতা"] in lines else True) or \
                        (extracted["Name"] != "Not found" and i > lines.index(extracted["Name"]) if extracted[
                                                                                                        "Name"] in lines else True) or \
                        (extracted["নাম"] != "Not found" and i > lines.index(extracted["নাম"]) if extracted[
                                                                                                      "নাম"] in lines else i > name_index):
                    extracted["মাতা"] = line.strip()
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


# Commented Code 2 Functions for Optional Use
# # Format OCR results for side-by-side comparison
# def format_ocr_results(tesseract_results, easyocr_results):
#     output = []
#     for field in fields_code2.keys():
#         tesseract_value = tesseract_results.get(field, "Not found")
#         easyocr_value = easyocr_results.get(field, "Not found")
#         output.append(f"tesseract -> {field}: {tesseract_value}   easy ocr -> {field}: {easyocr_value}")
#     return "\n".join(output)

# # Combine OCR results based on conditions
# def combine_ocr_results(tesseract_results, easyocr_results):
#     combined = {}
#     for field in fields_code2.keys():
#         tesseract_value = tesseract_results.get(field, "Not found")
#         easyocr_value = easyocr_results.get(field, "Not found")
#         # Condition 1: If both match, take Tesseract's value
#         if tesseract_value == easyocr_value:
#             combined[field] = tesseract_value
#         # Condition 2: If both are "Not found", use "Not found"
#         elif tesseract_value == "Not found" and easyocr_value == "Not found":
#             combined[field] = "Not found"
#         # Condition 3: If only one has data, take it
#         elif tesseract_value != "Not found" and easyocr_value == "Not found":
#             combined[field] = tesseract_value
#         elif tesseract_value == "Not found" and easyocr_value != "Not found":
#             combined[field] = easyocr_value
#         # Condition 4: If they differ, take the one with more words
#         else:
#             tesseract_words = len(tesseract_value.split())
#             easyocr_words = len(easyocr_value.split())
#             combined[field] = tesseract_value if tesseract_words >= easyocr_words else easyocr_value
#     output = [f"{field}: {value}" for field, value in combined.items()]
#     return "\n".join(output)

# Comparison Logic
def remove_special_chars(text, field):
    if text == "Not found" or not text:
        return text
    if field == "DateOfBirth":
        # Keep digits, letters, spaces, - / . for dates
        cleaned = re.sub(r"[^0-9A-Za-z\s\-/\.]", "", text).strip()
        # Replace - or / with space for consistency in comparison
        cleaned = re.sub(r"[-/]", " ", cleaned).strip()
        return re.sub(r"\s+", " ", cleaned).strip()
    if field == "IDNO":
        # Keep only digits
        return re.sub(r"[^0-9]", "", text).strip()
    # For other fields, keep letters, Bangla, spaces, and dots
    cleaned = re.sub(r"[^A-Za-z\s\.\u0980-\u09FF]", "", text).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_date_field(text):
    if text == "Not found" or not text:
        return text
    # Match valid date patterns: DD Month YYYY or DD-MM-YYYY or DD/MM/YYYY
    date_pattern = r'^(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\s*(.*)$'
    match = re.match(date_pattern, text, re.IGNORECASE)
    if match:
        # Return only the valid date part, excluding trailing characters
        return match.group(1).strip()
    return text


def clean_all_special_chars(text):
    if text == "Not found" or not text:
        return text
    # Keep letters, Bangla characters, digits, spaces, and dots; remove all else
    cleaned = re.sub(r"[^A-Za-z0-9\s\.\u0980-\u09FF]", "", text).strip()
    # Normalize multiple spaces to single space
    return re.sub(r"\s+", " ", cleaned).strip()


def is_single_letter(text):
    if not text or text == "Not found":
        return False
    # Check if text is a single character and is an English or Bangla letter
    if len(text) == 1:
        return bool(re.match(r'[A-Za-z\u0980-\u09FF]', text))
    return False


def clean_bangla_field(text):
    if text == "Not found" or not text:
        return text
    # Split text into words
    words = text.split()
    # Keep only words with no English letters (A-Za-z)
    cleaned_words = [word for word in words if not re.search(r'[A-Za-z]', word)]
    # Join back and normalize spaces
    cleaned = " ".join(cleaned_words).strip()
    return cleaned if cleaned else "Not found"


def clean_english_field(text):
    if text == "Not found" or not text:
        return text
    # Split text into words
    words = text.split()
    # Keep only words with no Bangla letters (উ0980-উ09FF)
    cleaned_words = [word for word in words if not re.search(r'[\u0980-\u09FF]', word)]
    # Join back and normalize spaces
    cleaned = " ".join(cleaned_words).strip()
    return cleaned if cleaned else "Not found"


def compare_outputs(t1, e1, t2, e2, t3, e3, field):
    # Normalize "No data found" to "Not found"
    t1 = "Not found" if t1 == "No data found" else t1
    e1 = "Not found" if e1 == "No data found" else e1
    t2 = "Not found" if t2 == "No data found" else t2
    e2 = "Not found" if e2 == "No data found" else e2
    t3 = "Not found" if t3 == "No data found" else t3
    e3 = "Not found" if e3 == "No data found" else e3

    # Clean DateOfBirth field to remove trailing characters
    if field == "DateOfBirth":
        t1 = clean_date_field(t1)
        e1 = clean_date_field(e1)
        t2 = clean_date_field(t2)
        e2 = clean_date_field(e2)
        t3 = clean_date_field(t3)
        e3 = clean_date_field(e3)

    # Clean all special characters except dots for all fields
    t1 = clean_all_special_chars(t1)
    e1 = clean_all_special_chars(e1)
    t2 = clean_all_special_chars(t2)
    e2 = clean_all_special_chars(e2)
    t3 = clean_all_special_chars(t3)
    e3 = clean_all_special_chars(e3)

    # Apply language-specific cleaning
    if field in ['নাম', 'পিতা', 'মাতা', 'স্বামী', 'স্ত্রী']:
        t1 = clean_bangla_field(t1)
        e1 = clean_bangla_field(e1)
        t2 = clean_bangla_field(t2)
        e2 = clean_bangla_field(e2)
        t3 = clean_bangla_field(t3)
        e3 = clean_bangla_field(e3)
    elif field == "Name":
        t1 = clean_english_field(t1)
        e1 = clean_english_field(e1)
        t2 = clean_english_field(t2)
        e2 = clean_english_field(e2)
        t3 = clean_english_field(t3)
        e3 = clean_english_field(e3)

    # Remove special characters (field-specific)
    t1_clean = remove_special_chars(t1, field)
    e1_clean = remove_special_chars(e1, field)
    t2_clean = remove_special_chars(t2, field)
    e2_clean = remove_special_chars(e2, field)
    t3_clean = remove_special_chars(t3, field)
    e3_clean = remove_special_chars(e3, field)

    outputs = [t1_clean, e1_clean, t2_clean, e2_clean, t3_clean, e3_clean]

    # Commented debug print statements for six outputs
    print("\n================= OCR COMPARISON =================")
    print(f"{'':<17} t1                        | e1                        | t2                        | e2                        | t3                        | e3")
    print(f"{'Raw Outputs':<17}: {t1:<25} | {e1:<25} | {t2:<25} | {e2:<25} | {t3:<25} | {e3}")
    print(f"{'Cleaned Outputs':<17}: {t1_clean:<25} | {e1_clean:<25} | {t2_clean:<25} | {e2_clean:<25} | {t3_clean:<25} | {e3_clean}")
    print("==================================================\n")

    # Condition 1: All "Not found"
    if all(val == "Not found" for val in outputs):
        return "Not found"

    # Filter out "Not found" for comparison
    valid_outputs = [val for val in outputs if val != "Not found"]
    valid_raw = [val for val in [t1, e1, t2, e2, t3, e3] if val != "Not found"]

    # Condition 8: One to five "Not found", one valid output
    if len(valid_outputs) == 1:
        if is_single_letter(valid_outputs[0]):
            return "Not found"
        return valid_outputs[0]

    # Condition 2 & 6: Two or more outputs match
    value_counts = {}
    for val in valid_outputs:
        value_counts[val] = value_counts.get(val, 0) + 1
    matching_values = [val for val, count in value_counts.items() if count >= 2]
    if matching_values:
        return matching_values[0]  # Take any matching value

    # Condition 3: Check for pairs (e.g., t1=e2, t2=e3, t3=e1)
    pairs = [
        (t1_clean, e2_clean, t1, e2),
        (t2_clean, e3_clean, t2, e3),
        (t3_clean, e1_clean, t3, e1)
    ]
    matching_pairs = [(v1, v2, r1, r2) for v1, v2, r1, r2 in pairs if v1 == v2 and v1 != "Not found"]
    if len(matching_pairs) >= 1:  # At least one pair matches
        best_value = None
        max_words = 0
        has_special = True
        for v1, _, r1, r2 in matching_pairs:
            words1 = len(v1.split())
            special1 = bool(re.search(r"[^A-Za-z\s\.\u0980-\u09FF0-9]", r1))  # Exclude dots
            special2 = bool(re.search(r"[^A-Za-z\s\.\u0980-\u09FF0-9]", r2))
            if not special1 and (not special2 or words1 >= max_words):
                best_value = v1
                max_words = words1
                has_special = special1
            elif not special2 and words1 >= max_words:
                best_value = v1
                max_words = words1
                has_special = special2
            elif words1 > max_words:
                best_value = v1
                max_words = words1
        if best_value:
            return best_value

    # Condition 7: Four or five "Not found", one or two have data
    not_found_count = outputs.count("Not found")
    if not_found_count in [4, 5] and len(valid_outputs) in [1, 2]:
        if len(valid_outputs) == 1:
            if is_single_letter(valid_outputs[0]):
                return "Not found"
            return valid_outputs[0]
        words1 = len(valid_outputs[0].split())
        words2 = len(valid_outputs[1].split())
        return valid_outputs[0] if words1 >= words2 else valid_outputs[1]

    # Condition 5: All unique, select one with 3 or 4 words
    unique_outputs = list(set(valid_outputs))
    if len(unique_outputs) == len(valid_outputs):
        for val in unique_outputs:
            word_count = len(val.split())
            if word_count in [3, 4]:
                return val

    # Fallback: Take the value with the most words
    max_words = 0
    best_val = valid_outputs[0] if valid_outputs else "Not found"
    for val in valid_outputs:
        words = len(val.split())
        if words > max_words:
            max_words = words
            best_val = val
    return best_val


# Main Processing Function
def process_image(image_path):
    # Code 1 Processing
    img = Image.open(image_path)
    tesseract_text1 = pytesseract.image_to_string(img, lang='ben+eng')
    tesseract_text1 = clean_header_text(tesseract_text1)
    tesseract_results1 = extract_fields_code1(tesseract_text1)
    tesseract_results1 = infer_name_from_lines(tesseract_text1, tesseract_results1)

    results = get_easyocr_text(image_path)
    easyocr_text1 = results
    easyocr_text1 = clean_header_text(easyocr_text1)
    easyocr_results1 = extract_fields_code1(easyocr_text1)
    easyocr_results1 = infer_name_from_lines(easyocr_text1, easyocr_results1)

    # Code 2 Processing with Preprocessing
    img_cv2 = cv2.imread(image_path)
    rotated_img = dskew(img_cv2)
    preprocessed_img, _ = preprocess_before_crop(rotated_img)
    tesseract_text2 = get_tesseract_ocr(preprocessed_img)
    tesseract_results2 = extract_fields_code2(tesseract_text2)

    easyocr_text2 = get_easyocr_text(preprocessed_img)
    easyocr_results2 = extract_fields_code2(easyocr_text2)

    # Code 3 Processing with Preprocessing just rotate
    img3 = cv2.imread(image_path)
    rotated_img2 = dskew(img3)
    rotated_img2 = crop_image(rotated_img2)

    rotated_img3 = Image.fromarray(rotated_img2)
    tesseract_text3 = pytesseract.image_to_string(rotated_img3, lang='ben+eng')
    tesseract_text3 = clean_header_text(tesseract_text3)
    tesseract_results3 = extract_fields_code1(tesseract_text3)
    tesseract_results3 = infer_name_from_lines(tesseract_text3, tesseract_results3)

    results = get_easyocr_text(rotated_img2)
    easyocr_text3 = results
    easyocr_text3 = clean_header_text(easyocr_text3)
    easyocr_results3 = extract_fields_code1(easyocr_text3)
    easyocr_results3 = infer_name_from_lines(easyocr_text3, easyocr_results3)

    # Optional: Uncomment to display Code 2's side-by-side comparison and merged results
    # print("\nCode 2 Individual Results:")
    # print(format_ocr_results(tesseract_results2, easyocr_results2))
    # print("\nCode 2 Combined Results:")
    # print(combine_ocr_results(tesseract_results2, easyocr_results2))

    # Combine Results
    final_results = {}
    for field in fields_code1:
        t1 = tesseract_results1.get(field, "Not found")
        e1 = easyocr_results1.get(field, "Not found")
        t2 = tesseract_results2.get(field, "Not found")
        e2 = easyocr_results2.get(field, "Not found")
        t3 = tesseract_results3.get(field, "Not found")
        e3 = easyocr_results3.get(field, "Not found")
        final_results[field] = compare_outputs(t1, e1, t2, e2, t3, e3, field)

    # Print results
    print("\nFinal Combined Results:")
    for field, value in final_results.items():
        print(f"{field}: {value}")

    return final_results


# Example Usage
image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/5.jpg"
final_results = process_image(image_path)