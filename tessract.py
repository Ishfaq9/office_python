import pytesseract
from PIL import Image
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#
# img = cv2.imread("C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/cropped.jpg")
# img_pil = Image.fromarray(img)
# text = pytesseract.image_to_string(img_pil, lang='Bengali+eng')
# print("Tesseract OCR Result:\n", text)



img = Image.open("C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_5.jpg")
text = pytesseract.image_to_string(img, lang='ben+eng')  # 'ben' = Bengali
print("Tesseract OCR Result:\n", text)


#### for all images loop
# import pytesseract
# from PIL import Image
# import cv2
# import os
#
#
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# # Folder containing images
# folder_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/"
#
# # Supported image extensions
# image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
#
# # Loop through each file in the folder
# for filename in os.listdir(folder_path):
#     if any(filename.lower().endswith(ext) for ext in image_extensions):
#         image_path = os.path.join(folder_path, filename)
#         print(f"\nProcessing image: {filename}")
#
#         # Read and convert the image
#         img = cv2.imread(image_path)
#         if img is None:
#             print("Could not read image.")
#             continue
#         img_pil = Image.fromarray(img)
#
#         # OCR
#         text = pytesseract.image_to_string(img_pil, lang='Bengali+eng')
#         print("Tesseract OCR Result:\n", text)