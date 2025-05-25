import pytesseract
from PIL import Image
import easyocr
import re
import string

img = "C:/Users/ishfaq.rahman/Desktop/1.jpg"
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
    'ID NO': r'ID\s*NO[:：]?\s*(\d{8,20})'
}

# Function to check if a value contains special characters (excluding space and Bengali characters)
def has_special_char(value):
    # You may adjust this regex to your needs
    return bool(re.search(r"[^\w\s\u0980-\u09FF]", value))



# Function to extract fields from OCR text using regex patterns
def extract_fields(text, fields):
    extracted = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            extracted[key] = value
        else:
            extracted[key] = "No data found"
    return extracted


# # Function to compare the results and return the better result based on conditions
# def compare_results(tesseract_value, easyocr_value):
#     # Condition 1: If both values match, return the same value (take either one)
#     if tesseract_value == easyocr_value:
#         return tesseract_value
#
#     # Condition 2: If both are "No data found", return that
#     elif tesseract_value == "No data found" and easyocr_value == "No data found":
#         return "No data found"
#
#     # Condition 3: If they don't match, compare word count and return the one with more than one word
#     else:
#         tesseract_words = len(tesseract_value.split())
#         easyocr_words = len(easyocr_value.split())
#
#         # Return the value with more than one word
#         if tesseract_words > 1 and easyocr_words == 1:
#             return tesseract_value
#         elif easyocr_words > 1 and tesseract_words == 1:
#             return easyocr_value
#         else:
#             # If both have more than one word, return the one with more words
#             return tesseract_value if tesseract_words >= easyocr_words else easyocr_value


# --- Tesseract OCR ---
tesseract_image_path = img
img = Image.open(tesseract_image_path)
tesseract_text = pytesseract.image_to_string(img, lang='ben+eng')

# Extract fields from Tesseract output
tesseract_results = extract_fields(tesseract_text, fields)

# --- EasyOCR ---
easyocr_image_path = img
reader = easyocr.Reader(['en', 'bn'])
results = reader.readtext(easyocr_image_path)
easyocr_text = "\n".join([text for (_, text, _) in results])

# Extract fields from EasyOCR output
easyocr_results = extract_fields(easyocr_text, fields)


# --- Combine and select best result based on rules ---
final_results = {}
for key in fields.keys():
    t_val = tesseract_results[key]
    e_val = easyocr_results[key]

    # Condition 1: If both match
    if t_val == e_val:
        final_results[key] = t_val

    # Condition 2: If both are 'No data found'
    elif t_val == "No data found" and e_val == "No data found":
        final_results[key] = "No data found"

    # # Condition 4: If one has special chars and other doesn't
    # elif has_special_char(t_val) and not has_special_char(e_val):
    #     final_results[key] = e_val
    # elif has_special_char(e_val) and not has_special_char(t_val):
    #     final_results[key] = t_val

    # Condition 3: Pick the one with more words
    else:
        t_words = len(t_val.split())
        e_words = len(e_val.split())
        final_results[key] = t_val if t_words > e_words else e_val

# --- Output ---
print("\nFinal Results:")
for key, value in final_results.items():
    print(f"{key}: {value}")

#--- Debugging: Print Raw OCR Outputs ---
print("\n--- Tesseract Raw Output ---")
print(tesseract_text)

print("\n--- EasyOCR Raw Output ---")
print(easyocr_text)

# --- Combine and Print Results ---
# print("\nValid Data Results:")
# for key in fields.keys():
#     tesseract_field = tesseract_results[key]
#     easyocr_field = easyocr_results[key]
#     final_result = compare_results(tesseract_field, easyocr_field)
#
#     # Only print fields with valid data (not None or "No data found")
#     if final_result != "No data found":
#         print(f"{key}: {final_result}")
