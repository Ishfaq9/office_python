from paddleocr import PaddleOCR
import re

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

#ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_db_box_thresh=0.4)

# Run OCR on image
results = ocr.ocr("C:/Users/ishfaq.rahman/Desktop/NID Images/6.jpg", cls=True)
#results = ocr.ocr("C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/cropped.jpg", cls=True)

# Join all detected text into one string
all_text = ' '.join([line[1][0] for block in results for line in block])

# Show raw OCR output
print("----- RAW TEXT -----")
print(all_text)

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


