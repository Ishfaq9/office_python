import keras_ocr
import cv2
import pytesseract
from matplotlib import pyplot as plt

# Optional: specify tesseract path if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Bangla language support in Tesseract

# Step 1: Load image
image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_2.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Text Detection using Keras-OCR
pipeline = keras_ocr.pipeline.Pipeline()
prediction_groups = pipeline.recognize([image_rgb])

# Visualize detections
keras_ocr.tools.drawAnnotations(image_rgb, prediction_groups[0])
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Step 3: Extract detected boxes and crop each for OCR
results = []
for box, _ in prediction_groups[0]:
    # Get box coordinates and crop
    x0, y0 = box[0]
    x2, y2 = box[2]
    cropped = image[int(y0):int(y2), int(x0):int(x2)]

    # Step 4: Run Tesseract OCR for Bangla on cropped box
    text = pytesseract.image_to_string(cropped, 'ben')
    results.append(text.strip())

# Final output
print("🔍 Bangla Text Detected:")
for i, txt in enumerate(results, 1):
    print(f"{i}. {txt}")
