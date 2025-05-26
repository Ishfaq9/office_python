import pytesseract
from PIL import Image
import easyocr
import re

# Image and Tesseract path
img_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_1.png"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define fields and regex
fields = {
    'নাম': r'নাম[:：]?\s*([^\n:：]+)',
    'Name': r'Name[:：]?\s*([^\n:：]+)',
    'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',
    'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',
    'স্বামী': r'স্বামী[:：]?\s*([^\n:：]+)',
    'স্ত্রী': r'স্ত্রী[:：]?\s*([^\n:：]+)',
    'Date of Birth': r'Date of Birth[:：]?\s*([^\n:：]+)',
    #'ID NO': r'(?:ID\s*NO|NID\s*No\.?|NID\s*NO)[:：]?\s*(\d{8,20})',
    'ID NO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]?\s*([\d ]{8,30})'
}

def infer_name_from_lines(text, extracted_fields):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if "Name" in line:
            if extracted_fields.get('নাম') in [None, "", "No data found"] and i > 0:
                extracted_fields['নাম'] = lines[i - 1]
            if extracted_fields.get('Name') in [None, "", "No data found"] and i + 1 < len(lines):
                extracted_fields['Name'] = lines[i + 1]
    return extracted_fields

def clean_header_text(text):
    keywords_to_remove = [
        "বাংলাদেশ সরকার",
        "জাতীয় পরিচয়",
        "জাতীয় পরিচয়",
        "National ID",
        "Government of the People's Republic"
    ]

    cleaned_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()
        if not any(keyword in line_stripped for keyword in keywords_to_remove):
            cleaned_lines.append(line_stripped)

    return "\n".join(cleaned_lines)

    # # Optional: remove multiple spaces or blank lines
    # lines = [line.strip() for line in text.splitlines() if line.strip()]
    # return "\n".join(lines)


# Function to extract fields using regex
# def extract_fields(text, fields):
#     extracted = {}
#     for key, pattern in fields.items():
#         match = re.search(pattern, text, re.IGNORECASE)
#         extracted[key] = match.group(1).strip() if match else "No data found"
#     return extracted

def extract_fields(text, fields):
    extracted = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if key == 'ID NO':
                value = value.replace(" ", "")  # Remove spaces from NID
            extracted[key] = value
        else:
            extracted[key] = "No data found"
    return extracted

# Tesseract OCR
img = Image.open(img_path)
tesseract_text = pytesseract.image_to_string(img, lang='ben+eng')
print("result=",tesseract_text)
# Extract fields from Tesseract output
tesseract_text = clean_header_text(tesseract_text)
tesseract_results = extract_fields(tesseract_text, fields)
tesseract_results = infer_name_from_lines(tesseract_text, tesseract_results)

# EasyOCR
reader = easyocr.Reader(['en', 'bn'])
results = reader.readtext(img_path)
print(results)
easyocr_text = "\n".join([text for (_, text, _) in results])

# Extract fields from EasyOCR output
easyocr_text = clean_header_text(easyocr_text)
easyocr_results = extract_fields(easyocr_text, fields)
easyocr_results = infer_name_from_lines(easyocr_text, easyocr_results)

# Merge logic
final_results = {}
#print("\nFinal Results:")
for key in fields:
    t_val = tesseract_results[key]
    e_val = easyocr_results[key]

    if t_val == "No data found" and e_val == "No data found":
        final = "No data found"
    elif t_val == e_val:
        final = t_val
    elif t_val != "No data found" and e_val == "No data found":
        final = t_val
    elif e_val != "No data found" and t_val == "No data found":
        final = e_val
    else:
        # Both have different non-empty data
        t_words = len(t_val.strip().split())
        e_words = len(e_val.strip().split())
        final = t_val if t_words >= e_words else e_val

    final_results[key] = final
    print(f"{key}: {final}")
