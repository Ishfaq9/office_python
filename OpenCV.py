import cv2
import numpy as np
import pytesseract

# Set path to tesseract.exe if not in system PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # Load the image
# image_path = "C:/Users/ishfaq.rahman/Desktop/5.jpg"
# image = cv2.imread(image_path)
#
# # Resize if needed (optional for large images)
# # image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
#
# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Apply thresholding for better text clarity
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# # OCR using pytesseract (both Bengali and English)
# custom_config = r'--oem 3 --psm 6'
# extracted_text = pytesseract.image_to_string(thresh, lang='ben+eng', config=custom_config)
#
# print("----- Raw Extracted Text -----\n")
# print(extracted_text)
#
# # Now extract specific fields using regex or heuristics
# def extract_nid_data(text):
#     data = {
#         "Name": None,
#         "Father's Name": None,
#         "Mother's Name": None,
#         "Date of Birth": None,
#         "NID Number": None
#     }
#
#     # Common Bengali markers
#     lines = text.split('\n')
#     for line in lines:
#         if 'নাম' in line and data["Name"] is None:
#             data["Name"] = line.split(':')[-1].strip()
#         elif 'পিতার নাম' in line or 'পিতা' in line:
#             data["Father's Name"] = line.split(':')[-1].strip()
#         elif 'মাতার নাম' in line or 'মাতা' in line:
#             data["Mother's Name"] = line.split(':')[-1].strip()
#         elif 'জন্ম তারিখ' in line or 'তারিখ' in line:
#             data["Date of Birth"] = re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', line)
#             if data["Date of Birth"]:
#                 data["Date of Birth"] = data["Date of Birth"].group()
#         else:
#             # Try catching NID numbers in 10-17 digit range
#             nid_match = re.findall(r'\b\d{10,17}\b', line)
#             if nid_match and data["NID Number"] is None:
#                 data["NID Number"] = nid_match[0]
#
#     return data
#
# # Extract structured data
# structured_data = extract_nid_data(extracted_text)
#
# print("\n----- Extracted Fields -----")
# for key, value in structured_data.items():
#     print(f"{key}: {value}")



# Optional: Set the path to tesseract.exe (Windows)

#
# def preprocess_nid(image_path, save_path='processed_nid.jpg'):
#     # Step 1: Read image
#     image = cv2.imread(image_path)
#     orig = image.copy()
#     h, w = image.shape[:2]
#
#     # Step 2: Resize to standard size
#     image = cv2.resize(image, (1000, int(1000 * h / w)))
#
#     # Step 3: Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Step 4: Denoise
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Step 5: Detect edges
#     edged = cv2.Canny(blur, 30, 150)
#
#     # Step 6: Find contours to locate the NID card
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
#
#     for cnt in contours:
#         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#         if len(approx) == 4:  # found rectangle
#             screen_cnt = approx
#             break
#     else:
#         screen_cnt = np.array([[[0, 0]], [[0, h]], [[w, h]], [[w, 0]]])
#
#     # Step 7: Perspective transform (crop NID card)
#     def order_points(pts):
#         pts = pts.reshape(4, 2)
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
#         return rect
#
#     def four_point_transform(image, pts):
#         rect = order_points(pts)
#         (tl, tr, br, bl) = rect
#
#         widthA = np.linalg.norm(br - bl)
#         widthB = np.linalg.norm(tr - tl)
#         maxWidth = max(int(widthA), int(widthB))
#
#         heightA = np.linalg.norm(tr - br)
#         heightB = np.linalg.norm(tl - bl)
#         maxHeight = max(int(heightA), int(heightB))
#
#         dst = np.array([
#             [0, 0],
#             [maxWidth - 1, 0],
#             [maxWidth - 1, maxHeight - 1],
#             [0, maxHeight - 1]], dtype="float32")
#
#         M = cv2.getPerspectiveTransform(rect, dst)
#         warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#         return warped
#
#     cropped = four_point_transform(image, screen_cnt)
#
#     # Step 8: Convert to grayscale again after crop
#     gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#
#     # Step 9: Sharpen
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5,-1],
#                        [0, -1, 0]])
#     sharpened = cv2.filter2D(gray_crop, -1, kernel)
#
#     # Step 10: Thresholding
#     thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)
#
#     # Step 11: Save final preprocessed image
#     cv2.imwrite(save_path, thresh)
#
#     return save_path
#
#
# # Run preprocessing
# processed_image_path = preprocess_nid("C:/Users/ishfaq.rahman/Desktop/6.jpg")
#
# # Run OCR on processed image (Bengali + English)
# custom_config = r'--oem 3 --psm 6 -l ben+eng'
# text = pytesseract.image_to_string(processed_image_path, config=custom_config)
#
# print("OCR Output:")
# print(text)

