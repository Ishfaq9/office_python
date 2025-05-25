# import pytesseract
# from PIL import Image
# import easyocr
# import re
# from tabulate import tabulate
#
# # Set Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# # Load images
# img = "C:/Users/ishfaq.rahman/Desktop/6.jpg"
# tesseract_img = Image.open(img)
# easyocr_img_path = img
#
# # Run Tesseract OCR
# tesseract_text = pytesseract.image_to_string(tesseract_img, lang='ben+eng')
#
# # Run EasyOCR
# reader = easyocr.Reader(['en', 'bn'])
# easyocr_results = reader.readtext(easyocr_img_path)
# easyocr_text = '\n'.join([text for (_, text, _) in easyocr_results])
#
# # Fields to extract
# fields = {
#     'নাম': r'নাম[:：]?\s*([^\n]+)',
#     'Name': r'Name[:：]?\s*([A-Z\s]+)',
#     'পিতা': r'পিতা[:：]?\s*([^\n]+)',
#     'মাতা': r'মাতা[:：]?\s*([^\n]+)',
#     'Date of Birth': r'Date of Birth[:：]?\s*(\d{2} [A-Za-z]{3,} \d{4})',
#     'ID NO': r'ID\s*NO[:：]?\s*(\d{10,})'
# }
#
# def extract_field(text, pattern):
#     match = re.search(pattern, text, re.IGNORECASE)
#     return match.group(1).strip() if match else "No data found"
#
# # Prepare data for table
# table_data = []
# for label, pattern in fields.items():
#     easy_val = extract_field(easyocr_text, pattern)
#     tess_val = extract_field(tesseract_text, pattern)
#     table_data.append([label, easy_val, tess_val])
#
# # Print formatted table
# print(tabulate(table_data, headers=["Model", "EasyOCR", "Tesseract OCR"], tablefmt="grid"))



import pytesseract
from PIL import Image
import easyocr
import re

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load images
img = "C:/Users/ishfaq.rahman/Desktop/6.jpg"
tesseract_img = Image.open(img)
easyocr_img_path = img

# Run Tesseract OCR
tesseract_text = pytesseract.image_to_string(tesseract_img, lang='ben+eng')
print("Tesseract OCR Result:\n", tesseract_text)

# Run EasyOCR
reader = easyocr.Reader(['en', 'bn'])
easyocr_results = reader.readtext(easyocr_img_path)
easyocr_text = '\n'.join([text for (_, text, _) in easyocr_results])
print("\nEasyOCR Result:\n", easyocr_text)

# Fields to extract
# fields = {
#     'নাম': r'নাম[:：]?\s*([^\n:：]+)',
#     'Name': r'Name[:：]?\s*([A-Z][A-Z\s]+)',
#     'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',
#     'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',
#     'Date of Birth': r'Date of Birth[:：]?\s*(\d{1,2} [A-Za-z]{3,} \d{4})',
#     'ID NO': r'ID\s*NO[:：]?\s*(\d{8,20})'
# }
fields = {
    'নাম': r'নাম[:：]?\s*([^\n:：]+)',
    'Name': r'Name[:：]?\s*([A-Z][A-Z\s]+)',
    'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',
    'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',
    'Date of Birth': r'Date of Birth[:：]?\s*(\d{1,2} [A-Za-z]{3,} \d{4})',
    'ID NO': r'ID\s*NO[:：]?\s*(\d{8,20})'
}

def extract_field(text, pattern):
    match = re.search(pattern, text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    else:
        return "No data found"


def clean_value(label, value):
    if value.lower().startswith(label.lower()):
        value = value[len(label):].strip(" :।-")

    # Remove unwanted "D" if it’s the result of a broken line
    if len(value) == 1 and value.isalpha():
        return "No data found"

    return value.strip()

# Extract data from both OCR engines
# print("\n--- Final Comparison ---")
# for label, pattern in fields.items():
#     easy_val = extract_field(easyocr_text, pattern)
#     tess_val = extract_field(tesseract_text, pattern)
#     print(f"{label} => EasyOCR: {easy_val}   Tesseract OCR: {tess_val}")
# for label, pattern in fields.items():
#     easy_val = extract_field(easyocr_text, pattern)
#     tess_val = extract_field(tesseract_text, pattern)
#     print(f"{label}: {easy_val}, {label}: {tess_val}")
# def clean_value(label, value):
#     if value.lower().startswith(label.lower()):
#         return value[len(label):].strip(" :।-")
#     return value.strip()

for label, pattern in fields.items():
    easy_val = extract_field(easyocr_text, pattern)
    tess_val = extract_field(tesseract_text, pattern)

    easy_val = clean_value(label, easy_val)
    tess_val = clean_value(label, tess_val)

    # Only print if at least one value is found
    if easy_val != "No data found" or tess_val != "No data found":
        print(f"{label}: {easy_val}, {label}: {tess_val}")




