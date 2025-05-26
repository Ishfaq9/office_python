import numpy as np
import pytesseract
from PIL import Image
import easyocr
import re
import cv2
from github_image_process_Tesse import dskew, preprocess_before_crop

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
reader = easyocr.Reader(['en', 'bn'], gpu=False)

# Code 1 Regex Patterns
fields_code1 = {
    'নাম': r'নাম[:：]?\s*([^\n:：]+)',
    'Name': r'Name[:：]?\s*([^\n:：]+)',
    'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',
    'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',
    'স্বামী': r'স্বামী[:：]?\s*([^\n:：]+)',
    'স্ত্রী': r'স্ত্রী[:：]?\s*([^\n:：]+)',
    'DateOfBirthh': r'Date of Birth[:：]?\s*([^\n:：]+)',
    'IDNO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]?\s*([\d ]{8,30})'
}

# Code 2 Regex Patterns
fields_code2 = {
    'নাম': r'নাম[:：]?\s*([^\n:：]+)',
    'Name': r'Name[:：]?\s*([^\n:：]+)',
    'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',
    'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',
    'স্বামী': r'(?:স্বামী|স্বা[:;মী-]*|husband|sami)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্ত্রী|Date|ID)',
    'স্ত্রী': r'(?:স্ত্রী|স্ত্র[:;ী-]*|wife|stri)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|Date|ID)',
    'DateOfBirth': r'(?:Date of Birth|DOB|Date|Birth)[:;\s-]*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|স্ত্রী|ID)',
    'IDNO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]?\s*([\d ]{8,30})'
}


# Code 1 Functions
# ✅ Preprocessing function (fixed version)
def preprocess_before_crop_Sharp(scan_path):
    # Sharpening
    adjusted_image = cv2.convertScaleAbs(scan_path, alpha=1.2, beta=20)
    kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted_image, -1, kernel)



    return sharpened

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
    id_no_pattern = r"(ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)[:：]?\s*([\d ]{8,30})"
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

def compare_outputs(t1, e1, t2, e2, t3, e3, field):
    # Normalize "No data found" to "Not found"
    t1 = "Not found" if t1 == "No data found" else t1
    e1 = "Not found" if e1 == "No data found" else e1
    t2 = "Not found" if t2 == "No data found" else t2
    e2 = "Not found" if e2 == "No data found" else e2
    t3 = "Not found" if t3 == "No data found" else t3
    e3 = "Not found" if e3 == "No data found" else e3

    # Clean Date of Birth field to remove trailing characters
    # Clean Date of Birth field to remove trailing characters
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
            special1 = bool(re.search(r"[^A-Za-z\s\.\u0980-\u09FF0-9\-/]", r1))
            special2 = bool(re.search(r"[^A-Za-z\s\.\u0980-\u09FF0-9\-/]", r2))
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
    rotated_img2 = preprocess_before_crop_Sharp(rotated_img2)
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
#image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_3.png"
final_results = process_image(image_path)