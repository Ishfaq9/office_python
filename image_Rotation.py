import cv2
import pytesseract
import os
import numpy as np
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def correct_image_alignment(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize image to simulate higher DPI (better for OCR)
    scale_percent = 150  # Resize scale
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Preprocess: Convert to grayscale & enhance text
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Reduce noise
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)

    # Step 1: Try to detect and fix rotation using Tesseract
    try:
        osd = pytesseract.image_to_osd(thresh, output_type=Output.DICT)
        rotation_angle = osd['rotate']
        print(f"[INFO] Detected rotation: {rotation_angle} degrees")

        if rotation_angle == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_angle == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotation_angle == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    except pytesseract.TesseractError as e:
        print(f"[WARNING] Tesseract OSD failed: {e}")
        print("[INFO] Skipping orientation correction.")

    # Step 2: Skew correction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        print("[WARNING] No text found for skew correction.")
        deskewed_image = image
    else:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        print(f"[INFO] Skew angle detected: {angle:.2f} degrees")

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed_image = cv2.warpAffine(image, matrix, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

    # Step 3: Save to "corrected" subfolder
    base_dir = os.path.dirname(image_path)
    output_dir = os.path.join(base_dir, 'corrected')
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"corrected_{filename}")
    cv2.imwrite(output_path, deskewed_image)

    print(f"[DONE] Corrected image saved at: {output_path}")
    return output_path


# Example usage
source_image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/12.jpg"
correct_image_alignment(source_image_path)
