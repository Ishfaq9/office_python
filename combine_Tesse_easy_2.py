import pytesseract
from PIL import Image
import easyocr
import re

img = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/enhanced images/7.jpg"
# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define fields with regex patterns
fields = {
    'নাম': r'নাম[:：]?\s*([^\n:：]+)',
    'Name': r'Name[:：]?\s*([^\n:：]+)',
    'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',
    'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',
    'স্বামী': r'স্বামী[:：]?\s*([^\n:：]+)',
    'Date of Birth': r'Date of Birth[:：]?\s*([^\n:：]+)',
    #'ID NO': r'(?:ID\s*NO|NID\s*No\.?|NID\s*NO)[:：]?\s*(\d{8,20})',
    'ID NO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No)\s*[:：]?\s*([\d ]{8,30})'
}


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


def infer_name_from_lines(text, extracted_fields):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if "Name" in line:
            if extracted_fields.get('নাম') in [None, "", "No data found"] and i > 0:
                extracted_fields['নাম'] = lines[i - 1]
            if extracted_fields.get('Name') in [None, "", "No data found"] and i + 1 < len(lines):
                extracted_fields['Name'] = lines[i + 1]
    return extracted_fields



# Function to extract fields from OCR text using regex patterns
# def extract_fields(text, fields):
#     extracted = {}
#     for key, pattern in fields.items():
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             value = match.group(1).strip()
#             extracted[key] = value
#         else:
#             extracted[key] = "No data found"
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


# --- Tesseract OCR ---

tesseract_image_path = img
img = Image.open(tesseract_image_path)
tesseract_text = pytesseract.image_to_string(img, lang='ben+eng') #Bengali_best ben

# Extract fields from Tesseract output
#tesseract_text = clean_header_text(tesseract_text)
tesseract_results = extract_fields(tesseract_text, fields)
#tesseract_results = infer_name_from_lines(tesseract_text, tesseract_results)

# --- EasyOCR ---
easyocr_image_path = img
reader = easyocr.Reader(['en', 'bn'])
results = reader.readtext(easyocr_image_path)
easyocr_text = "\n".join([text for (_, text, _) in results])

# Extract fields from EasyOCR output
#easyocr_text = clean_header_text(easyocr_text)
easyocr_results = extract_fields(easyocr_text, fields)
#easyocr_results = infer_name_from_lines(easyocr_text, easyocr_results)



# --- Combine and Print Results ---
print("\nCombined Results:")
for key in fields.keys():
    tesseract_field = f"{key}: {tesseract_results[key]}"
    easyocr_field = f"{key}: {easyocr_results[key]}"
    print(f"{tesseract_field} , {easyocr_field}")

# Optional: Print raw OCR outputs for debugging
print("\nTesseract Raw Output:")
print(tesseract_text)
print("\nEasyOCR Raw Output:")
print(easyocr_text)