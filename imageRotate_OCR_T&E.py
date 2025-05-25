import cv2
import numpy as np
from math import atan, degrees, radians, sin, cos, fabs
import pytesseract
from PIL import Image
import easyocr
import re

img_path="C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_3.png"

class ImgCorrect():
    def __init__(self, img):
        self.img = img
        self.h, self.w, self.channel = self.img.shape
        # print("Original images h & w -> | w: ",self.w, "| h: ",self.h)
        if self.w <= self.h:
            self.scale = 700 / self.w
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.scale = 700 / self.h
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        #print("Resized Image by Padding and Scaling:")
        #plot_fig(self.img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def img_lines(self):
        print("Gray Image:")
        #plot_fig(self.gray)
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("bin",binary)
        print("Inverse Binary:")
        #plot_fig(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # rectangular structure
        # print("Kernel for dialation:")
        # print(kernel)
        binary = cv2.dilate(binary, kernel)  # dilate
        print("Dilated Binary:")
        #plot_fig(binary)
        edges = cv2.Canny(binary, 50, 200)
        print("Canny edged detection:")
        #plot_fig(edges)

        # print("Edge 1: ")
        # cv2.imshow("edges", edges)

        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        # print(self.lines)
        if self.lines is None:
            print("Line segment not found")
            return None

        lines1 = self.lines[:, 0, :]  # Extract as 2D
        # print(lines1)
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        print("Probabilistic Hough Lines:")
        #plot_fig(imglines)
        return imglines

    def search_lines(self):
      lines = self.lines[:, 0, :]  # extract as 2D

      number_inexist_k = 0
      sum_pos_k45 = number_pos_k45 = 0
      sum_pos_k90 = number_pos_k90 = 0
      sum_neg_k45 = number_neg_k45 = 0
      sum_neg_k90 = number_neg_k90 = 0
      sum_zero_k = number_zero_k = 0

      for x in lines:
          if x[2] == x[0]:
              number_inexist_k += 1
              continue
          #print(degrees(atan((x[3] - x[1]) / (x[2] - x[0]))), "pos:", x[0], x[1], x[2], x[3], "Slope:",(x[3] - x[1]) / (x[2] - x[0]))
          degree = degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
          # print("Degree or Slope of detected lines : ",degree)
          if 0 < degree < 45:
              number_pos_k45 += 1
              sum_pos_k45 += degree
          if 45 <= degree < 90:
              number_pos_k90 += 1
              sum_pos_k90 += degree
          if -45 < degree < 0:
              number_neg_k45 += 1
              sum_neg_k45 += degree
          if -90 < degree <= -45:
              number_neg_k90 += 1
              sum_neg_k90 += degree
          if x[3] == x[1]:
              number_zero_k += 1

      max_number = max(number_inexist_k, number_pos_k45, number_pos_k90, number_neg_k45,number_neg_k90, number_zero_k)
      # print("Num of lines in different Degree range ->")
      # print("Not a Line: ",number_inexist_k, "| 0 to 45: ",number_pos_k45, "| 45 to 90: ",number_pos_k90, "| -45 to 0: ",number_neg_k45, "| -90 to -45: ",number_neg_k90, "| Line where y1 equals y2 :",number_zero_k)

      if max_number == number_inexist_k:
          return 90
      if max_number == number_pos_k45:
          return sum_pos_k45 / number_pos_k45
      if max_number == number_pos_k90:
          return sum_pos_k90 / number_pos_k90
      if max_number == number_neg_k45:
          return sum_neg_k45 / number_neg_k45
      if max_number == number_neg_k90:
          return sum_neg_k90 / number_neg_k90
      if max_number == number_zero_k:
          return 0

    def rotate_image(self, degree):
        """
        Positive angle counterclockwise rotation
        :param degree:
        :return:
        """
        # print("degree:", degree)
        if -45 <= degree <= 0:
            degree = degree  # #negative angle clockwise
        if -90 <= degree < -45:
            degree = 90 + degree  # positive angle counterclockwise
        if 0 < degree <= 45:
            degree = degree  # positive angle counterclockwise
        if 45 < degree <= 90:
            degree = degree - 90  # negative angle clockwise
        print("DSkew angle: ", degree)

        # degree = degree - 90
        height, width = self.img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(
            cos(radians(degree))))  # This formula refers to the previous content
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        # print("Height :",height)
        # print("Width :",width)
        # print("HeightNew :",heightNew)
        # print("WidthNew :",widthNew)

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # rotate degree counterclockwise
        # print("Mat Rotation (Before): ",matRotation)
        matRotation[0, 2] += (widthNew - width) / 2
        # Because after rotation, the origin of the coordinate system is the upper left corner of the new image, so it needs to be converted according to the original image
        matRotation[1, 2] += (heightNew - height) / 2
        # print("Mat Rotation (After): ",matRotation)

        # Affine transformation, the background color is filled with white
        imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        # Padding
        pad_image_rotate = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(0, 255, 0))
        #plot_fig(pad_image_rotate)

        return imgRotation


def dskew(img):
    #img_loc = line_path + img
    #im = cv2.imread(img_loc)
    im=img
    # Padding
    bg_color = [255, 255, 255]
    pad_img = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value=bg_color)

    imgcorrect = ImgCorrect(pad_img)
    lines_img = imgcorrect.img_lines()
    # print(type(lines_img))

    if lines_img is None:
        rotate = imgcorrect.rotate_image(0)
    else:
        degree = imgcorrect.search_lines()
        rotate = imgcorrect.rotate_image(degree)


    return rotate


image = cv2.imread(img_path)
rotate = dskew(image)

cv2.bitwise_not(rotate,rotate)
#-------------------- ocr part ---------------------

# Image and Tesseract path
img_path = rotate
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define fields and regex
fields = {
    'নাম': r'নাম[:：]?\s*([^\n:：]+)',
    'Name': r'Name[:：]?\s*([^\n:：]+)',
    'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',
    'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',
    'স্বামী': r'স্বামী[:：]?\s*([^\n:：]+)',
    'স্ত্রী': r'স্ত্রী[:：]?\s*([^\n:：]+)',
    'Date of Birth': r'Date of Birth[:：]?\s*([^\n:：]+)',
    #'ID NO': r'(?:ID\s*NO|NID\s*No\.?|NID\s*NO)[:：]?\s*(\d{8,20})',
    'ID NO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No)\s*[:：]?\s*([\d ]{8,30})'
}

def infer_name_from_lines(text, extracted_fields):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if "Name" in line:
            if extracted_fields.get('নাম') in [None, "", "No data found"] and i > 0:
                extracted_fields['নাম'] = lines[i - 1]
            if extracted_fields.get('Name') in [None, "", "No data found"] and i + 1 < len(lines):
                extracted_fields['Name'] = lines[i + 1]
    return extracted_fields

def clean_header_text(text):
    keywords_to_remove = [
        "বাংলাদেশ সরকার",
        "জাতীয় পরিচয়",
        "জাতীয় পরিচয়",
        "National ID",
        "Government of the People's Republic"
    ]

    cleaned_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()
        if not any(keyword in line_stripped for keyword in keywords_to_remove):
            cleaned_lines.append(line_stripped)

    return "\n".join(cleaned_lines)

    # # Optional: remove multiple spaces or blank lines
    # lines = [line.strip() for line in text.splitlines() if line.strip()]
    # return "\n".join(lines)


# Function to extract fields using regex
# def extract_fields(text, fields):
#     extracted = {}
#     for key, pattern in fields.items():
#         match = re.search(pattern, text, re.IGNORECASE)
#         extracted[key] = match.group(1).strip() if match else "No data found"
#     return extracted

def extract_fields(text, fields):
    extracted = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if key == 'ID NO':
                value = value.replace(" ", "")  # Remove spaces from NID
            extracted[key] = value
        else:
            extracted[key] = "No data found"
    return extracted

# Tesseract OCR
#img = Image.open(img_path)
img = Image.fromarray(rotate)

tesseract_text = pytesseract.image_to_string(img, lang='ben+eng')

# Extract fields from Tesseract output
tesseract_text = clean_header_text(tesseract_text)
tesseract_results = extract_fields(tesseract_text, fields)
tesseract_results = infer_name_from_lines(tesseract_text, tesseract_results)

# EasyOCR
reader = easyocr.Reader(['en', 'bn'])
results = reader.readtext(rotate)
easyocr_text = "\n".join([text for (_, text, _) in results])

# Extract fields from EasyOCR output
easyocr_text = clean_header_text(easyocr_text)
easyocr_results = extract_fields(easyocr_text, fields)
easyocr_results = infer_name_from_lines(easyocr_text, easyocr_results)

# Merge logic
final_results = {}
#print("\nFinal Results:")
for key in fields:
    t_val = tesseract_results[key]
    e_val = easyocr_results[key]

    if t_val == "No data found" and e_val == "No data found":
        final = "No data found"
    elif t_val == e_val:
        final = t_val
    elif t_val != "No data found" and e_val == "No data found":
        final = t_val
    elif e_val != "No data found" and t_val == "No data found":
        final = e_val
    else:
        # Both have different non-empty data
        t_words = len(t_val.strip().split())
        e_words = len(e_val.strip().split())
        final = t_val if t_words >= e_words else e_val

    final_results[key] = final
    print(f"{key}: {final}")
