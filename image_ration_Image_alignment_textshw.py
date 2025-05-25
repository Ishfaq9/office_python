import cv2
import numpy as np
import os
from pathlib import Path


def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # Convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Use ORB to detect keypoints and extract local invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # Match features using brute-force Hamming matcher
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # Sort matches and keep only the best ones
    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # Extract matched keypoints
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # Compute homography and align
    (H, _) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    return aligned


def correct_skew(image, draw_angle_text=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]

    print(angle)

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle


    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if draw_angle_text:
        cv2.putText(rotated, f"Angle: {angle:.2f} degrees", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return rotated


def process_image(input_image_path, template_image_path):
    # Load input and template images
    input_image = cv2.imread(input_image_path)
    template_image = cv2.imread(template_image_path)

    if input_image is None or template_image is None:
        print("Error: Could not load input or template image.")
        return

    # Align and correct skew
    aligned_image = align_images(input_image, template_image)
    corrected_image = correct_skew(aligned_image)

    # Save result to a new folder under source image folder
    src_folder = Path(input_image_path).parent
    save_folder = src_folder / "aligned_outputs"
    save_folder.mkdir(parents=True, exist_ok=True)

    output_path = save_folder / Path(input_image_path).name
    cv2.imwrite(str(output_path), corrected_image)

    print(f"[✔] Saved aligned & corrected image to: {output_path}")


# Example usage
# Replace these with your actual paths
source_image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/12.jpg"
template_image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/test.jpg"
process_image(source_image_path, template_image_path)
