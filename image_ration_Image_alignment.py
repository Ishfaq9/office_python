import cv2
import pytesseract
import numpy as np
import os
from pytesseract import Output
import imutils

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#  Step 1: Align image to template using ORB + Homography
def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descsA, descsB, None)
    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    return aligned


#  Step 2: Full image correction with alignment, rotation, skew fix
def correct_image_alignment(image_path, template_path):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    if image is None or template is None:
        raise FileNotFoundError("Image or template not found.")

    # ️ Resize image to simulate high DPI
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    #  Align to template
    aligned_image = align_images(image, template)

    # ➡ Convert to grayscale & threshold
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)

    #  Orientation fix with Tesseract
    try:
        osd = pytesseract.image_to_osd(thresh, output_type=Output.DICT)
        rotation_angle = osd['rotate']
        print(f"[INFO] Detected rotation: {rotation_angle}°")

        if rotation_angle == 90:
            aligned_image = cv2.rotate(aligned_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_angle == 180:
            aligned_image = cv2.rotate(aligned_image, cv2.ROTATE_180)
        elif rotation_angle == 270:
            aligned_image = cv2.rotate(aligned_image, cv2.ROTATE_90_CLOCKWISE)
    except pytesseract.TesseractError as e:
        print(f"[WARNING] Tesseract OSD failed: {e}")
        print("[INFO] Skipping orientation correction.")

    #  Skew correction
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        print(f"[INFO] Skew angle detected: {angle:.2f}°")

        (h, w) = aligned_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected_image = cv2.warpAffine(aligned_image, M, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
    else:
        corrected_image = aligned_image
        print("[WARNING] Skew correction skipped (no text found).")

    #  Save to corrected subfolder
    output_dir = os.path.join(os.path.dirname(image_path), "corrected")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"corrected_{filename}")
    cv2.imwrite(output_path, corrected_image)
    print(f"[DONE] Corrected image saved at: {output_path}")
    return output_path


#  Example usage
source_image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/7.jpg"
template_image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/test.jpg"
correct_image_alignment(source_image_path, template_image_path)
