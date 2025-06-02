import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pytesseract
import easyocr
import re
import json
import sys
import cv2
import numpy as np
from math import atan, degrees, radians, sin, cos, fabs
import cv2
import os
import pytesseract
from PIL import Image
import easyocr
import re
import json
import sys
import cv2
import numpy as np
from math import atan, degrees, radians, sin, cos, fabs



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
        #print("Gray Image:")
        #plot_fig(self.gray)
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("bin",binary)
        #print("Inverse Binary:")
        #plot_fig(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # rectangular structure
        # print("Kernel for dialation:")
        # print(kernel)
        binary = cv2.dilate(binary, kernel)  # dilate
        #print("Dilated Binary:")
        #plot_fig(binary)
        edges = cv2.Canny(binary, 50, 200)
        #print("Canny edged detection:")
        #plot_fig(edges)

        # print("Edge 1: ")
        # cv2.imshow("edges", edges)

        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        # print(self.lines)
        if self.lines is None:
            #print("Line segment not found")
            return None

        lines1 = self.lines[:, 0, :]  # Extract as 2D
        # print(lines1)
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #print("Probabilistic Hough Lines:")
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
        #print("DSkew angle: ", degree)

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

# ✅ Preprocessing function (fixed version)
def preprocess_before_crop(scan_path):
    #original_image = cv2.imread(scan_path)
    original_image = scan_path

    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Skip equalizeHist to avoid background merging problems

    # Initial Denoising
    #denoised = cv2.fastNlMeansDenoising(gray, None, h=20, templateWindowSize=7, searchWindowSize=21)

    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # Bilateral filter
    bilateral_filtered = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

    # Contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(bilateral_filtered)

    # Blur
    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)

    # Adaptive Threshold
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    return gray, original_image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Folder containing images
folder_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/"

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(folder_path, filename)
        #print(f"\nProcessing image: {filename}")

        # Read and convert the image
        img = cv2.imread(image_path)
        rotate = dskew(img)
        preprocessed_image, original_image = preprocess_before_crop(rotate)
        #img = cv2.imread(image_path)
        if preprocessed_image is None:
            #print("Could not read image.")
            continue
        #img_pil = Image.fromarray(img)
        img_pil = Image.fromarray(preprocessed_image)

        # OCR
        text = pytesseract.image_to_string(img_pil, lang='Bengali+eng')
        print("Tesseract OCR Result:\n", text)
