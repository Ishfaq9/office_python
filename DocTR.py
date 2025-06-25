import os

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def extract_text_and_tables(image_path):
    try:
        doc = DocumentFile.from_images(image_path)
        predictor = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
        result = predictor(doc)

        raw_text = ""
        for page in result.pages:
            for block in page.blocks:
                if block.artefacts:  # Check for table-like structures
                    for artefact in block.artefacts:
                        raw_text += f"Table cell: {artefact.value}\n"
                for line in block.lines:
                    for word in line.words:
                        raw_text += word.value + " "
                    raw_text += "\n"

        return raw_text.strip()

    except Exception as e:
        return f"Error: {str(e)}"


# Example usage
folder_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/"

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(folder_path, filename)
        print(f"\nProcessing image: {filename}")

        # Read and convert the image
        extracted_text = extract_text_and_tables(image_path)
        print(extracted_text)



