import easyocr
import cv2
import numpy as np
import os

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'bn'])  # Bengali + English

# Load image
image_path = "C:/Users/ishfaq.rahman/Desktop/6.jpg"
img = cv2.imread(image_path)
if img is None:
    print(f"❌ Could not read image at {image_path}")
    exit()

# Resize
img = cv2.resize(img, (800, 600))

# Helper to run OCR and print result
def run_easyocr(image, step_name):
    results = reader.readtext(image)
    print(f"\n🧾 OCR Result - {step_name}:\n{'-' * (20 + len(step_name))}")
    for (_, text, prob) in results:
        print(f"{text}")
    if not results:
        print("[No text detected]")

# Step 1: Raw Image
run_easyocr(img, "Raw Image")

# Step 2: Brightness/contrast adjustment
alpha = 1.0
beta = -50
bright_contrast = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
run_easyocr(bright_contrast, "Brightness Adjusted")

# Step 3: Grayscale + manual threshold
gray = cv2.cvtColor(bright_contrast, cv2.COLOR_BGR2GRAY)
gray[gray > 150] = 255
run_easyocr(gray, "Grayscale + Manual Threshold")

# Step 4: Adaptive threshold
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 50
)
run_easyocr(adaptive_thresh, "Adaptive Threshold")

# Step 5: Light color filtering (make light pixels white)
filtered = np.copy(bright_contrast)
light_pixels = np.where(
    (filtered[:, :, 0] >= 100) & (filtered[:, :, 1] >= 100) & (filtered[:, :, 2] >= 100)
)
filtered[light_pixels] = (255, 255, 255)
filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
run_easyocr(filtered_gray, "After Light Color Filtering")

# Step 6: AutoCrop
# Use the adaptive threshold to find contours
contours, hierarchy = cv2.findContours(adaptive_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest bounding box
mx = (0, 0, 0, 0)
mx_area = 0
for cont in contours:
    x, y, w, h = cv2.boundingRect(cont)
    area = w * h
    if area > mx_area:
        mx = x, y, w, h
        mx_area = area

x, y, w, h = mx
cropped = img[y:y + h, x:x + w]
cv2.imwrite("Image_crop.jpg", cropped)  # optional save

# Optional: save image with bounding box drawn
cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)
cv2.imwrite("Image_cont.jpg", img)

# OCR after AutoCrop
run_easyocr(cropped, "After AutoCrop (Largest Contour)")
