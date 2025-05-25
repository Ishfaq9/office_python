import cv2
import numpy as np
import pytesseract
import easyocr
import re
import  os
import matplotlib.pyplot as plt

# tressract
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Apply thresholding to enhance text
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     return thresh
#
# def extract_text(image):
#     text = pytesseract.image_to_string(image, lang='eng+ben')  # Support for English and Bengali
#     return text
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # Load and preprocess image
# image_path = "C:/Users/ishfaq.rahman/Desktop/5.jpg"
# processed_image = preprocess_image(image_path)
# extracted_text = extract_text(processed_image)
# print(extracted_text)



#easy ocr
# def extract_text_easyocr(image_path):
#     reader = easyocr.Reader(['en', 'bn'])  # English and Bengali
#     result = reader.readtext(image_path,detail=0)
#     return result
#     #return ' '.join([text[1] for text in result])
#
# extracted_text = extract_text_easyocr("C:/Users/ishfaq.rahman/Desktop/7.jpg")
# print("\n", "\n".join(extracted_text))





# Set your Tesseract cmd path if not added to PATH (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directory where images are located
image_dir = "C:/Users/ishfaq.rahman/Desktop/"  # Use your actual image directory path
output_file = 'ocr_output.txt'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Make sure output file is empty before starting
open(output_file, 'w').close()

# List images in the folder
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(image_files)} images.")

# Loop through images
for image_file in image_files:
    print(f"\nProcessing: {image_file}")
    img_path = os.path.join(image_dir, image_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read {img_path}")
        continue

    # Resize if very large
    if img.shape[0] > 1000 or img.shape[1] > 1000:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 85, 11
    )

    # Show image (optional for debugging)
    # plt.imshow(processed_img, cmap='gray')
    # plt.title(image_file)
    # plt.axis('off')
    # plt.show()

    # OCR - assuming English and Bengali
    try:
        text = pytesseract.image_to_string(processed_img, lang='ben+eng')
    except pytesseract.TesseractError as e:
        print(f"OCR failed: {e}")
        continue

    print("OCR Result:\n", text)

    # Save to output file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n--- OCR for {image_file} ---\n{text}\n")

print(f"\nOCR completed. Results saved to {output_file}")










