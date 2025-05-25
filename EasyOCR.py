import easyocr
import cv2
import matplotlib.pyplot as plt

# Load image
image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_5.jpg"
reader = easyocr.Reader(['en', 'bn'])  # Bengali + English

results = reader.readtext(image_path)

# Print and visualize
for (bbox, text, prob) in results:
    print(f"{text}")


