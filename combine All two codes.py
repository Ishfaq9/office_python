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
    'DateOfBirth': r'Date of Birth[:：]?\s*([^\n:：]+)',
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
    #reader = easyocr.Reader(['en', 'bn'], gpu=False)
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
        return re.sub(r"[^0-9A-Za-z\s\-/]", "", text).strip()
    if field == "IDNO":
        return re.sub(r"[^0-9]", "", text).strip()
    return re.sub(r"[^A-Za-z\s\.\u0980-\u09FF]", "", text).strip()


def compare_outputs(t1, e1, t2, e2, field):
    # Normalize "No data found" to "Not found"
    t1 = "Not found" if t1 == "No data found" else t1
    e1 = "Not found" if e1 == "No data found" else e1
    # Remove special characters
    t1_clean = remove_special_chars(t1, field)
    e1_clean = remove_special_chars(e1, field)
    t2_clean = remove_special_chars(t2, field)
    e2_clean = remove_special_chars(e2, field)
    outputs = [t1_clean, e1_clean, t2_clean, e2_clean]

    # print("\n================= OCR COMPARISON =================")
    print(f"{'':<17} t1                        | e1                        | t2                        | e2")
    print(f"{'Raw Outputs':<17}: {t1:<25} | {e1:<25} | {t2:<25} | {e2}")
    print(f"{'Cleaned Outputs':<17}: {t1_clean:<25} | {e1_clean:<25} | {t2_clean:<25} | {e2_clean}")
    print("==================================================\n")


    # Condition 1: All "Not found"
    if all(val == "Not found" for val in outputs):
        return "Not found"

    # Condition 8: Three "Not found", one has data
    not_found_count = outputs.count("Not found")
    if not_found_count == 3:
        for val in outputs:
            if val != "Not found":
                return val

    # Condition 2 & 6: Two or three outputs match
    value_counts = {}
    for val in outputs:
        if val != "Not found":
            value_counts[val] = value_counts.get(val, 0) + 1
    matching_values = [val for val, count in value_counts.items() if count >= 2]
    if matching_values:
        return matching_values[0]  # Take any matching value

    # Condition 3: Two pairs match (e.g., t1=e2 and t2=e1)
    if t1_clean == e2_clean and t2_clean == e1_clean and t1_clean != t2_clean:
        t1_words = len(t1_clean.split()) if t1_clean != "Not found" else 0
        t2_words = len(t2_clean.split()) if t2_clean != "Not found" else 0
        t1_has_special = bool(re.search(r"[^A-Za-z\s\.\u0980-\u09FF0-9\-/]", t1))
        t2_has_special = bool(re.search(r"[^A-Za-z\s\.\u0980-\u09FF0-9\-/]", t2))
        if not t1_has_special and (t2_has_special or t1_words >= t2_words):
            return t1_clean
        return t2_clean

    # Condition 7: Two "Not found", two have data
    if not_found_count == 2:
        valid_outputs = [val for val in outputs if val != "Not found"]
        if len(valid_outputs) == 2:
            words1 = len(valid_outputs[0].split())
            words2 = len(valid_outputs[1].split())
            return valid_outputs[0] if words1 >= words2 else valid_outputs[1]

    # Condition 5: All unique, select one with 3 or 4 words
    unique_outputs = list(set(outputs) - {"Not found"})
    if len(unique_outputs) == len([val for val in outputs if val != "Not found"]):
        for val in unique_outputs:
            word_count = len(val.split())
            if word_count in [3, 4]:
                return val

    # Fallback: Take the value with the most words
    max_words = 0
    best_val = "Not found"
    for val in outputs:
        if val != "Not found":
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
        final_results[field] = compare_outputs(t1, e1, t2, e2, field)

    # Print results
    print("\nFinal Combined Results:")
    for field, value in final_results.items():
        print(f"{field}: {value}")

    return final_results


# Example Usage
image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/5.jpg"
final_results = process_image(image_path)