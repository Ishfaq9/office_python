import pytesseract
import easyocr
import cv2 as cv
import os
import re
from PIL import Image
from image_enhancement import image_enhancement

# --- Input image ---
original_img_path = "C:/Users/ishfaq.rahman/Desktop/5.jpg"
filename = os.path.basename(original_img_path)
enhanced_folder = os.path.join(os.path.dirname(original_img_path), "Enhanced")
enhanced_img_path = os.path.join(enhanced_folder, filename)

# --- Create output folder if not exists ---
os.makedirs(enhanced_folder, exist_ok=True)

# --- Read original image and enhance using GHE ---
input_img = cv.imread(original_img_path)
# ie = image_enhancement.IE(input_img, color_space='RGB')
# output_img = ie.GHE()  # You can try AHE(), GC(), etc. instead

ie = image_enhancement.IE(input_img, color_space='RGB')
output_img = ie.BBHE()

# --- Save the enhanced image ---
cv.imwrite(enhanced_img_path, output_img)
print(f"Enhanced image saved at: {enhanced_img_path}")

# --- Set Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Define regex patterns ---
fields = {
    'নাম': r'নাম[:：]?\s*([^\n:：]+)',
    'Name': r'Name[:：]?\s*([^\n:：]+)',
    'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',
    'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',
    'স্বামী': r'স্বামী[:：]?\s*([^\n:：]+)',
    'Date of Birth': r'Date of Birth[:：]?\s*([^\n:：]+)',
    'ID NO': r'ID\s*NO[:：]?\s*(\d{8,20})'
}

# --- Extract text with regex ---
def extract_fields(text, fields):
    extracted = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE)
        extracted[key] = match.group(1).strip() if match else "No data found"
    return extracted

# --- Tesseract OCR ---
img = Image.open(enhanced_img_path)
tesseract_text = pytesseract.image_to_string(img, lang='ben+eng')
tesseract_results = extract_fields(tesseract_text, fields)

# --- EasyOCR ---
reader = easyocr.Reader(['en', 'bn'])
results = reader.readtext(enhanced_img_path)
easyocr_text = "\n".join([text for (_, text, _) in results])
easyocr_results = extract_fields(easyocr_text, fields)

# --- Final merge logic ---
final_results = {}
print("\nFinal Results:")
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
        t_words = len(t_val.strip().split())
        e_words = len(e_val.strip().split())
        final = t_val if t_words >= e_words else e_val

    final_results[key] = final
    print(f"{key}: {final}")
