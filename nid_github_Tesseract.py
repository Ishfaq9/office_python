import pytesseract
import cv2
import os
import shutil
import numpy as np

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_dir = r"C:/Program Files/Tesseract-OCR/tessdata/"

# Verify Bengali.traineddata
traineddata = "Bengali.traineddata"
if traineddata not in os.listdir(tessdata_dir):
    print(f"❌ {traineddata} is missing in {tessdata_dir}")
else:
    print(f"✅ Found {traineddata} in {tessdata_dir}")

# Load image
img_path = "C:/Users/ishfaq.rahman/Desktop/6.jpg"
img = cv2.imread(img_path)
if img is None:
    print(f"❌ Could not read image from: {img_path}")
    exit()

# Step 1: Raw image OCR
text_raw = pytesseract.image_to_string(img, lang='Bengali+eng')
print("\n🧾 OCR Result - Raw Image:\n----------------------------\n", text_raw)

# Step 2: Resize
img_resized = cv2.resize(img, (800, 600))
text_resized = pytesseract.image_to_string(img_resized, lang='Bengali+eng')
print("\n🧾 OCR Result - Resized (800x600):\n------------------------------------\n", text_resized)

# Step 3: Brightness/contrast adjustment
alpha = 1.0
beta = -50
bright_contrast = np.clip(alpha * img_resized + beta, 0, 255).astype(np.uint8)
cv2.imwrite("bright_contrast.jpg", bright_contrast)
text_bright = pytesseract.image_to_string(bright_contrast, lang='Bengali+eng')
print("\n🧾 OCR Result - Brightness Adjusted:\n--------------------------------------\n", text_bright)

# Step 4: Grayscale + Manual threshold
gray = cv2.cvtColor(bright_contrast, cv2.COLOR_BGR2GRAY)
gray[gray > 150] = 255
cv2.imwrite("gray_thresh.jpg", gray)
text_gray = pytesseract.image_to_string(gray, lang='Bengali+eng')
print("\n🧾 OCR Result - Grayscale + Manual Threshold:\n-----------------------------------------------\n", text_gray)

# Step 5: Adaptive threshold
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 50
)
cv2.imwrite("adaptive_thresh.jpg", adaptive_thresh)
text_adaptive = pytesseract.image_to_string(adaptive_thresh, lang='Bengali+eng')
print("\n🧾 OCR Result - Adaptive Threshold:\n-------------------------------------\n", text_adaptive)

# Step 6: Color filtering (remove light background)
filtered = np.copy(bright_contrast)
light_pixels = np.where(
    (filtered[:, :, 0] >= 100) & (filtered[:, :, 1] >= 100) & (filtered[:, :, 2] >= 100)
)
filtered[light_pixels] = (255, 255, 255)
filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
cv2.imwrite("filtered.jpg", filtered_gray)
text_filtered = pytesseract.image_to_string(filtered_gray, lang='Bengali+eng')
print("\n🧾 OCR Result - After Light Color Filtering:\n---------------------------------------------\n", text_filtered)
