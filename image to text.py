# import pytesseract
# from PIL import Image

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# img = Image.open("C:/Users/ishfaq.rahman/Desktop/6.jpg")
# text = pytesseract.image_to_string(img, lang='ben+eng')  # 'ben' = Bengali
# print("Tesseract OCR Result:\n", text)

# import easyocr
#
# reader = easyocr.Reader(['bn', 'en'])
# results = reader.readtext("C:/Users/ishfaq.rahman/Desktop/6.jpg", detail=0)
# print("EasyOCR Result:\n", "\n".join(results))


#working good so far
# import cv2
# import easyocr
#
# # Load image with OpenCV
# img = cv2.imread("C:/Users/ishfaq.rahman/Desktop/4.jpg")
#
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply thresholding (binary)
# _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# # Save or process directly
# cv2.imwrite("processed.jpg", thresh)
#
# # OCR
# reader = easyocr.Reader(['bn', 'en'])
# results = reader.readtext("processed.jpg", detail=0)
# print("Improved OCR Result:\n", "\n".join(results))


# import pytesseract
# import cv2
#
# # Set the path to the Tesseract executable (if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# # Load the image
# image = cv2.imread("C:/Users/ishfaq.rahman/Desktop/6.jpg")
#
# # Preprocess the image (as described above)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# # Configure Tesseract to recognize Bangla
# custom_config = r'--oem 3 --psm 6 -l ben'
#
# # Extract text using pytesseract
# text = pytesseract.image_to_string(thresh, config=custom_config)
# print(text)


import pytesseract
import cv2
import os
import shutil

tessdata_dir = "C:/Program Files/Tesseract-OCR/tessdata/"
if os.path.exists(tessdata_dir):
    print("Existing trained data files:", os.listdir(tessdata_dir))
else:
    print(f"{tessdata_dir} does not exist. Skipping list.")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # Copy traineddata file (if you have it locally)
src = "C:/Program Files/Tesseract-OCR/tessdata/Bengali.traineddata"  # Update path based on your project
# Avoid SameFileError by checking if source and destination are different
if os.path.exists(src):
    dest = os.path.join(tessdata_dir, "Bengali.traineddata")
    if not os.path.samefile(src, dest):
        shutil.copy(src, dest)
        print("Bengali.traineddata copied.")
    else:
        print("Source and destination are the same file. Skipping copy.")
else:
    print("Bengali.traineddata not found in input path.")

# # Load and preprocess image
img_path = "C:/Users/ishfaq.rahman/Desktop/5.jpg"  # Update to local image path
#img_path = "C:/Users/ishfaq.rahman/Desktop/img_bn_3.png"  # Update to local image path
img = cv2.imread(img_path)
#
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # OCR
    text = pytesseract.image_to_string(thresh, lang='Bengali_best+en')  # Use 'ben' for Bengali
    print("Extracted Text:\n", text)
else:
    print(f"Image not found at path: {img_path}")


























