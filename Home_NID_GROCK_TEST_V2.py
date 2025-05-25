import re
from datetime import datetime
import os
import re
import logging

import cv2
from PIL import Image
import pytesseract
import easyocr
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Get Tesseract OCR text from image
def get_tesseract_ocr(image_path):
    img = Image.open(image_path)
    #img = Image.fromarray(image_path)
    ocr_text = pytesseract.image_to_string(img, lang='ben+eng')
    return ocr_text


reader = easyocr.Reader(['en', 'bn'], gpu=False)
# Get EasyOCR text from image
def get_easyocr_text(image_path):
    # reader = easyocr.Reader(['en', 'bn'], gpu=False)
    results = reader.readtext(image_path)
    # with open(r'C:\Users\Ishfaq\Desktop\imge.txt', 'r', encoding='utf-8') as file:
    #     results = file.read().strip()
    # Convert EasyOCR results (list of tuples) to single text string
    ocr_text = "\n".join([text for _, text, _ in results])

    return ocr_text


def clean_text(text):
    """Clean raw OCR text aggressively to remove garbage."""
    garbage_patterns = [
        r'গণপ্রজাতন্ত্রী বাংলাদেশ সরকার.*',
        r'Government of.*',
        r'National ID Card.*',
        r'জাতীয় পরিচয় পত্র.*',
        r'জাতীয় পস প.*',
        r'\b(Imhbin|Ma|Recxraccr|LErlzশt|সাইফলইষ্য্য|সইফলইষযয)\b'
    ]
    for pattern in garbage_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s:.-/\u0980-\u09FF]', '', text)
    return text


def clean_field_value(value, field):
    """Clean field value to remove English special characters, except dot in Name field."""
    if field == "Name":
        return re.sub(r'[^\w\s.]', '', value)
    elif field == "Date of Birth":
        return re.sub(r'[^\w\s]', '', value)
    elif field == "ID NO":
        return re.sub(r'\D', '', value)
    else:
        return re.sub(r'[^\w\s\u0980-\u09FF]', '', value)


def parse_date(date_str):
    """Parse date into DD MMM YYYY format, handling split lines."""
    date_formats = [
        r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',  # e.g., 03 Feb 2004
        r'\d{1,2}-\d{1,2}-\d{4}',  # e.g., 03-02-2004
        r'\d{1,2}/\d{1,2}/\d{4}',  # e.g., 03/02/2004
        r'\b(\d{1,2})\s+(\d{4})\b'  # e.g., 11 1965
    ]
    months = {
        'jan': 'Jan', 'january': 'Jan', 'feb': 'Feb', 'february': 'Feb', 'mar': 'Mar', 'march': 'Mar',
        'apr': 'Apr', 'april': 'Apr', 'may': 'May', 'jun': 'Jun', 'june': 'Jun', 'jul': 'Jul', 'july': 'Jul',
        'aug': 'Aug', 'august': 'Aug', 'sep': 'Sep', 'september': 'Sep', 'oct': 'Oct', 'october': 'Oct',
        'nov': 'Nov', 'november': 'Nov', 'dec': 'Dec', 'december': 'Dec'
    }

    for fmt in date_formats:
        match = re.search(fmt, date_str, re.IGNORECASE)
        if match:
            date_part = match.group(0)
            if ' ' in date_part and not re.match(r'\b\d{1,2}\s+\d{4}\b', date_part):
                day, month, year = date_part.split()
                month = months.get(month.lower(), month[:3].capitalize())
                try:
                    datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
                    return f"{day} {month} {year}"
                except ValueError:
                    continue
            elif re.match(r'\b\d{1,2}\s+\d{4}\b', date_part):
                day, year = match.groups()
                month = 'Jan'
                for m in months.values():
                    if m.lower() in date_str.lower():
                        month = m
                        break
                try:
                    datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
                    return f"{day} {month} {year}"
                except ValueError:
                    continue
            else:
                day, month, year = re.split(r'[-/]', date_part)
                try:
                    dt = datetime.strptime(f"{day}-{month}-{year}", "%d-%m-%Y")
                    return dt.strftime("%d %b %Y")
                except ValueError:
                    continue
    return None


def extract_nid_fields(raw_text):
    """Extract NID fields from raw OCR text, handling missing labels and garbage."""
    cleaned_text = clean_text(raw_text)
    result = {
        "নাম": "Not found",
        "Name": "Not found",
        "পিতা": "Not found",
        "মাতা": "Not found",
        "স্বামী": "Not found",
        "স্ত্রী": "Not found",
        "Date of Birth": "Not found",
        "ID NO": "Not found"
    }
    # patterns = {
    #     "নাম": r'(?:নাম|নামর)[:\s]+([^\nপিতাNameমাতাস্বামীস্ত্রীDate of BirthID NO]+)',
    #     "Name": r'Name[:\s]+([^\nপিতানামমাতাস্বামীস্ত্রীDate of BirthID NO]+)',
    #     "পিতা": r'(?:পিতা|পত)[:\s]+([^\nনামNameমাতাস্বামীস্ত্রীDate of BirthID NO]+)',
    #     "মাতা": r'(?:মাতা|মত)[:\s]+([^\nনামNameপিতাস্বামীস্ত্রীDate of BirthID NO]+)',
    #     "স্বামী": r'স্বামী[:\s]+([^\nনামNameপিতামাতাস্ত্রীDate of BirthID NO]+)',
    #     "স্ত্রী": r'স্ত্রী[:\s]+([^\nনামNameপিতামাতাস্বামীDate of BirthID NO]+)',
    #     "ID NO": r'(?:ID NO|1D NO|NID No|NID NO|ID No)[:\s/]*(\d{10,17})|\b(\d{10,17})\b'
    # }
    patterns = {
        'নাম': r'নাম[:：]?\s*([^\n:：]+)',  # GitHub: Simple for clean labels
        'Name': r'Name[:：]?\s*([^\n:：]+)',  # GitHub: Simple for clean labels
        'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',  # GitHub: Simple for clean labels
        'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',  # GitHub: Simple for clean labels
        'স্বামী': r'(?:স্বামী|স্বা[:;মী-]*|husband|sami)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্ত্রী|Date|ID)',
        # Original: Handles OCR errors
        'স্ত্রী': r'(?:স্ত্রী|স্ত্র[:;ী-]*|wife|stri)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|Date|ID)',
        # Original: Handles OCR errors
      # Original: Strict formats
        'ID NO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]?\s*([\d ]{8,30})'
        # GitHub: Precise range
    }

    # Split text into lines for processing
    lines = cleaned_text.split('\n')
    current_field = None
    field_value = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for labeled fields
        for field, pattern in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = match.group(1) or match.group(2)
                if current_field and field_value:
                    result[current_field] = clean_field_value(' '.join(field_value).strip(), current_field)
                    field_value = []
                current_field = field
                field_value.append(value)
                break
        else:
            # Accumulate unlabeled lines for the current field
            if current_field and not re.search(r'(?:নাম|নামর|Name|পিতা|পত|মাতা|মত|স্বামী|স্ত্রী|Date of Birth|ID NO)',
                                               line, re.IGNORECASE):
                field_value.append(line)

    # Save the last field
    if current_field and field_value:
        result[current_field] = clean_field_value(' '.join(field_value).strip(), current_field)

    # Extract Date of Birth
    date_str = parse_date(cleaned_text)
    if date_str:
        result["Date of Birth"] = date_str

    # Handle unlabeled fields (পিতা, মাতা) after নাম/Name
    name_index = -1
    for i, line in enumerate(lines):
        if re.search(r'(?:নাম|নামর|Name)[:\s]+', line, re.IGNORECASE):
            name_index = i
            break

    if name_index != -1:
        potential_fields = []
        for i in range(name_index + 1, len(lines)):
            line = lines[i].strip()
            if re.search(r'(?:Date of Birth|ID NO|1D NO|NID No|NID NO|ID No|মাতা|মত|স্বামী|স্ত্রী)', line,
                         re.IGNORECASE):
                break
            if line and not re.search(r'(?:নাম|নামর|Name|পিতা|পত)[:\s]+', line, re.IGNORECASE):
                potential_fields.append(line)

        if len(potential_fields) >= 1 and result["পিতা"] == "Not found":
            result["পিতা"] = clean_field_value(potential_fields[0], "পিতা")
        if len(potential_fields) >= 2 and result["মাতা"] == "Not found":
            result["মাতা"] = clean_field_value(potential_fields[1], "মাতা")

    return result


def combine_results(tesseract_result, easyocr_result):
    """Combine Tesseract and EasyOCR results based on specified rules."""
    combined_result = {}
    fields = ["নাম", "Name", "পিতা", "মাতা", "স্বামী", "স্ত্রী", "Date of Birth", "ID NO"]

    for field in fields:
        t_val = tesseract_result[field]
        e_val = easyocr_result[field]

        # Rule 1: If both match, take Tesseract's value
        if t_val == e_val:
            combined_result[field] = t_val
        # Rule 2: If both are "Not found", use "Not found"
        elif t_val == "Not found" and e_val == "Not found":
            combined_result[field] = "Not found"
        # Rule 4: If one is "Not found", take the other
        elif t_val == "Not found":
            combined_result[field] = e_val
        elif e_val == "Not found":
            combined_result[field] = t_val
        # Rule 3: If neither is "Not found" and they don't match, take the one with more words
        else:
            t_words = len(t_val.split())
            e_words = len(e_val.split())
            combined_result[field] = e_val if e_words >= t_words else t_val

    return combined_result


def format_output(tesseract_result, easyocr_result, combined_result, image_name):
    """Format Tesseract, EasyOCR, and combined results."""
    fields = ["নাম", "Name", "পিতা", "মাতা", "স্বামী", "স্ত্রী", "Date of Birth", "ID NO"]
    output = [f"Processing image: {image_name}"]
    for field in fields:
        output.append(f"tesseract -> {field}: {tesseract_result[field]}   easy ocr -> {field}: {easyocr_result[field]}")
    output.append("\nCombined Results:")
    for field in fields:
        output.append(f"{field}: {combined_result[field]}")
    return "\n".join(output)


def process_nid_text(tesseract_text, easyocr_text, image_name):
    """Process Tesseract and EasyOCR text and return formatted output."""
    tesseract_result = extract_nid_fields(tesseract_text)
    easyocr_result = extract_nid_fields(easyocr_text)
    combined_result = combine_results(tesseract_result, easyocr_result)
    return format_output(tesseract_result, easyocr_result, combined_result, image_name)







# Example usage for NID_1.png
img = ("C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/cropped2.jpg")
tesseract_text = get_tesseract_ocr(img)
easyocr_text = get_easyocr_text(img)

print(process_nid_text(tesseract_text, easyocr_text, "ishfaq"))
print("-" * 50)