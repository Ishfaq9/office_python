import os
import warnings
from paddleocr import PaddleOCR


warnings.filterwarnings("ignore", category=UserWarning, module="paddle.utils.cpp_extension")

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_1.png"
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

results = ocr.ocr(image_path, cls=True)

# Correct way to extract text
all_text = ""
for line in results[0]:  # First (and only) image
    text = line[1][0]  # line[1] = [text, confidence]
    all_text += text + "\n"

print(all_text)

#all_text = ' '.join([line[1][0] for block in results for line in block]) if results and results[0] else ""

# Show raw OCR output
# print("----- RAW TEXT -----")
# print(all_text)

# # Extract fields using regex
# def extract_field(pattern, text):
#     match = re.search(pattern, text)
#     return match.group(1).strip() if match else None
#
# # Bengali & English patterns
# name_bn = extract_field(r'নাম[:\-]?\s*(মোঃ\s?[^\n]+)', all_text)
# name_en = extract_field(r'Name[:\-]?\s*(Md\.?\s+[A-Za-z\s]+)', all_text)
# father_bn = extract_field(r'পিতা[:\-]?\s*(মোঃ\s?[^\n]+)', all_text)
# mother_bn = extract_field(r'মাতা[:\-]?\s*(মোছা[:ঃ]?\s?[^\n]+)', all_text)
# dob = extract_field(r'Date of Birth[:\-]?\s*([0-9]{1,2} \w+ \d{4})', all_text)
# nid = extract_field(r'ID NO[:\-]?\s*(\d{10,20})', all_text)


