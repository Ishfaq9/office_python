import os
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Initialize the OCR predictor once to avoid reloading for each image
# Using 'db_resnet50' for detection and 'crnn_vgg16_bn' for recognition as per your original code.
# Consider using GPU if available for significant speedup.
predictor = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')

def extract_text_and_tables(image_path):
    # For single images, DocumentFile.from_images can directly take the path
    # No need to use cv2.imread and then pass the numpy array unless you're doing
    # pre-processing with OpenCV.
    doc = DocumentFile.from_images([image_path])
    result = predictor(doc)

    raw_text = [] # Use a list to build string for efficiency

    for page in result.pages:
        for block in page.blocks:
            if block.artefacts:  # Check for table-like structures
                for artefact in block.artefacts:
                    raw_text.append(f"Table cell: {artefact.value}\n")
            for line in block.lines:
                # Join words in a line directly for cleaner code and potentially faster
                # string concatenation.
                raw_text.append(" ".join([word.value for word in line.words]) + "\n")

    return "".join(raw_text).strip() # Join at the end

# Example usage
folder_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/"

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff') # Use a tuple for `any()`

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Using endswith with a tuple of extensions is more efficient
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(folder_path, filename)
        print(f"\nProcessing image: {filename}")

        #image_path = cv2.imread(image_path)
        extracted_text = extract_text_and_tables(image_path)
        print(extracted_text)