import easyocr
import cv2
import matplotlib.pyplot as plt

# Load image
image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_1.png"
reader = easyocr.Reader(['bn','en'])  # Bengali + English

results = reader.readtext(image_path)

# Print and visualize
for (bbox, text, prob) in results:
    print(f"{text}")


