import cv2
import easyocr
from PIL import Image
import pytesseract
import os
import re

# ---------- CONFIG ----------
image_path = "C:/Users/ishfaq.rahman/Desktop/4.jpg"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ----------------------------

# Load and preprocess image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Save for OCR processing
temp_filename = f"{os.getpid()}.png"
cv2.imwrite(temp_filename, thresh)

# OCR with Tesseract
tess_text = pytesseract.image_to_string(Image.open(temp_filename), lang='ben+eng')
os.remove(temp_filename)

# OCR with EasyOCR
reader = easyocr.Reader(['bn', 'en'])
easy_text_lines = reader.readtext(thresh, detail=0)

# ---------------------- Extraction Logic ----------------------

# From Tesseract (English fields)
name = re.search(r"Name:\s*(.+)", tess_text)
dob = re.search(r"Date of Birth:\s*(.+)", tess_text)
nid = re.search(r"ID\s*NO[: ]\s*(\d+)", tess_text)

# From EasyOCR (Bengali fields)
bangla_data = "\n".join(easy_text_lines)
bn_name = re.search(r"নাম[:\s]*([\u0980-\u09FF\s]+)", bangla_data)
bn_father = re.search(r"পিতা[:\s]*([\u0980-\u09FF\s]+)", bangla_data)
bn_mother = re.search(r"মাতা[:\s]*([\u0980-\u09FF\s]+)", bangla_data)

# ---------------------- Final Merged Output ----------------------

# Print in the specified format
print(f"নাম: {bn_name.group(1).strip() if bn_name else ''}")
print(f"পিতা: {bn_father.group(1).strip() if bn_father else ''}")
print(f"মাতা: {bn_mother.group(1).strip() if bn_mother else ''}")
print(f"Name: {name.group(1).strip() if name else ''}")
print(f"Date of Birth: {dob.group(1).strip() if dob else ''}")
print(f"ID NO: {nid.group(1).strip() if nid else ''}")
