import os
import re
import logging

import cv2
from PIL import Image
import pytesseract
import easyocr

from github_image_process_Tesse import dskew, preprocess_before_crop

# Configure logging


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Merged regex patterns combining GitHub simplicity and original robustness
patterns = {
    'নাম': r'নাম[:：]?\s*([^\n:：]+)',  # GitHub: Simple for clean labels
    'Name': r'Name[:：]?\s*([^\n:：]+)',  # GitHub: Simple for clean labels
    'পিতা': r'পিতা[:：]?\s*([^\n:：]+)',  # GitHub: Simple for clean labels
    'মাতা': r'মাতা[:：]?\s*([^\n:：]+)',  # GitHub: Simple for clean labels
    'স্বামী': r'(?:স্বামী|স্বা[:;মী-]*|husband|sami)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্ত্রী|Date|ID)',  # Original: Handles OCR errors
    'স্ত্রী': r'(?:স্ত্রী|স্ত্র[:;ী-]*|wife|stri)[:;\s-]*(.+?)(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|Date|ID)',  # Original: Handles OCR errors
    'Date of Birth': r'(?:Date of Birth|DOB|Date|Birth)[:;\s-]*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?=\n|$|নাম|Name|পিতা|মাতা|স্বামী|স্ত্রী|ID)',  # Original: Strict formats
    'ID NO': r'(?:ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)\s*[:：]?\s*([\d ]{8,30})'  # GitHub: Precise range
}


# Get Tesseract OCR text from image
def get_tesseract_ocr(image_path):
    #img = Image.open(image_path)
    img = Image.fromarray(image_path)
    ocr_text = pytesseract.image_to_string(img, lang='ben+eng')
    return ocr_text


reader = easyocr.Reader(['en', 'bn'], gpu=False)
# Get EasyOCR text from image
def get_easyocr_text(image_path):
    # reader = easyocr.Reader(['en', 'bn'], gpu=False)
    results = reader.readtext(image_path)
    # with open(r'C:\Users\Ishfaq\Desktop\imge.txt', 'r', encoding='utf-8') as file:
    #     results = file.read().strip()
    # Convert EasyOCR results (list of tuples) to single text string
    ocr_text = "\n".join([text for _, text, _ in results])

    return ocr_text


# Function to check for English alphabet letters
def contains_english(text):
    if not text or text == "Not found":
        return False
    return bool(re.search(r'[a-zA-Z]', text))

# Function to check for Bangla letters
def contains_bangla(text):
    if not text or text == "Not found":
        return False
    return bool(re.search(r'[\u0980-\u09FF]', text))


# Clean OCR text
def clean_ocr_text(ocr_text):
    # Split the text into lines
    lines = ocr_text.splitlines()

    # Find the index of the line containing 'নাম' or 'Name'
    name_index = -1
    for i, line in enumerate(lines):
        if 'নাম' in line:
            name_index = i
            break
        elif 'Name' in line and name_index == -1:
            name_index = i

    # Keep only lines from 'নাম' or 'Name' onward
    if name_index != -1:
        lines = lines[name_index:]

    # Join lines back to text for further processing
    ocr_text = "\n".join(lines)

    # Existing cleaning logic
    # Regex-based keyword removal
    keywords_to_remove = [
        r"গণপ্রজাতন্ত্রী বাংলাদেশ সরকার",
        r"গণপ্রজাতন্ত্রী সরকার",
        r"গণপ্রজাতন্ত্রী",
        r"বাংলাদেশ সরকার",
        r"Government of the People",
        r"National ID Card",
        r"জাতীয় পরিচয় পত্র",
        r"জাতীয় পরিচয়",
        r"20/05/2025 09:12"
    ]

    # Protect Date of Birth values
    dob_pattern = r"(Date of Birth|DOB|Date|Birth)[:：]?\s*(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    dob_matches = []

    def store_dob(match):
        dob_matches.append(match.group(0))
        return f"__DOB_{len(dob_matches) - 1}__"

    ocr_text = re.sub(dob_pattern, store_dob, ocr_text, flags=re.IGNORECASE)

    # Protect ID NO values
    id_no_pattern = r"(ID\s*NO|NID\s*No\.?|NIDNo|NID\s*NO|NID\s*No|ID\s*N0)[:：]?\s*([\d ]{8,30})"
    id_no_matches = []

    def store_id_no(match):
        id_no_matches.append(match.group(0))
        return f"__ID_NO_{len(id_no_matches) - 1}__"

    ocr_text = re.sub(id_no_pattern, store_id_no, ocr_text, flags=re.IGNORECASE)

    # Process each line individually
    lines = ocr_text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        # Remove keywords
        for keyword in keywords_to_remove:
            line = re.sub(keyword, "", line, flags=re.IGNORECASE)
        # Clean unwanted digits and brackets
        line = re.sub(r"[\[\]\(\)\{\}0-9]{3,}", "", line)
        # Normalize spaces within the line
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    # Restore Date of Birth and ID NO values
    ocr_text = "\n".join(cleaned_lines)
    for i, dob in enumerate(dob_matches):
        ocr_text = ocr_text.replace(f"__DOB_{i}__", dob)
    for i, id_no in enumerate(id_no_matches):
        ocr_text = ocr_text.replace(f"__ID_NO_{i}__", id_no)

    return ocr_text




# Merge fragmented lines
def merge_lines(ocr_text):
    lines = ocr_text.splitlines()
    merged_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i].strip()
        if re.match(r"^[^\x00-\x7F]+$", current_line) and len(current_line) < 10 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if (re.match(r"^[^\x00-\x7F]+$", next_line) and
                not any(re.search(pattern, next_line, re.IGNORECASE) for pattern in patterns.values())):
                merged_lines.append(current_line + " " + next_line)
                i += 2
                continue
        if re.match(r"^\d{1,2}$", current_line) and i + 1 < len(lines) and re.match(r"^\d{4}$", lines[i + 1]):
            merged_lines.append(current_line + " Jan " + lines[i + 1])
            i += 2
            continue
        merged_lines.append(current_line)
        i += 1
    return "\n".join(merged_lines)



# Clean Bangla name
def clean_bangla_name(name):
    if not name or name == "Not found":
        return name
    cleaned = re.sub(r"[^\u0980-\u09FF\s]", "", name).strip()
    return re.sub(r"\s+", " ", cleaned).strip()

# Clean English name
def clean_english_name(name):
    if not name or name == "Not found":
        return name
    cleaned = re.sub(r"[^A-Za-z\s\.]", "", name).strip()
    return re.sub(r"\s+", " ", cleaned).strip()

# Clean Date of Birth
def clean_date_of_birth(date):
    if not date or date == "Not found":
        return date
    cleaned = re.sub(r"[^0-9A-Za-z\s\-/]", "", date).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if re.match(r"^\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", cleaned, re.IGNORECASE):
        year = int(re.search(r"\d{4}", cleaned).group())
        if 1900 <= year <= 2025:
            return cleaned
    return "Invalid"

# Clean ID NO
def clean_id_no(id_no):
    if not id_no or id_no == "Not found":
        return id_no
    cleaned = re.sub(r"[^0-9]", "", id_no).strip()
    if re.match(r"^\d{10}$|^\d{13}$|^\d{17}$", cleaned):
        return cleaned
    return "Invalid"

# Extract fields from OCR text
def extract_fields(ocr_text):
    extracted = {key: "Not found" for key in patterns}

    ocr_text = clean_ocr_text(ocr_text)
    ocr_text = merge_lines(ocr_text)

    for key, pattern in patterns.items():
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()

    lines = ocr_text.splitlines()
    name_index = -1
    for i, line in enumerate(lines):
        if not any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns.values()):
            if re.match(r"[^\x00-\x7F]+", line) and extracted["নাম"] == "Not found":
                extracted["নাম"] = line.strip()
                name_index = i
            elif re.match(r"[A-Za-z\s\.]+", line) and extracted["Name"] == "Not found" and (name_index == -1 or i > name_index):
                extracted["Name"] = line.strip()
            elif (re.match(r"[^\x00-\x7F]+", line) and extracted["পিতা"] == "Not found"):
                # Check if line comes after Name or নাম, if they exist
                if (extracted["Name"] != "Not found" and i > lines.index(extracted["Name"]) if extracted["Name"] in lines else True) or \
                   (extracted["নাম"] != "Not found" and i > lines.index(extracted["নাম"]) if extracted["নাম"] in lines else i > name_index):
                    extracted["পিতা"] = line.strip()
            elif (re.match(r"[^\x00-\x7F]+", line) and extracted["মাতা"] == "Not found" and extracted["পিতা"] != "Not found"):
                # Check if line comes after পিতা or Name/নাম
                if (extracted["পিতা"] != "Not found" and i > lines.index(extracted["পিতা"]) if extracted["পিতা"] in lines else True) or \
                   (extracted["Name"] != "Not found" and i > lines.index(extracted["Name"]) if extracted["Name"] in lines else True) or \
                   (extracted["নাম"] != "Not found" and i > lines.index(extracted["নাম"]) if extracted["নাম"] in lines else i > name_index):
                    extracted["মাতা"] = line.strip()

    # Clean extracted fields
    extracted["নাম"] = clean_bangla_name(extracted["নাম"])
    extracted["পিতা"] = clean_bangla_name(extracted["পিতা"])
    extracted["মাতা"] = clean_bangla_name(extracted["মাতা"])
    extracted["স্বামী"] = clean_bangla_name(extracted["স্বামী"])
    extracted["স্ত্রী"] = clean_bangla_name(extracted["স্ত্রী"])
    extracted["Name"] = clean_english_name(extracted["Name"])
    extracted["Date of Birth"] = clean_date_of_birth(extracted["Date of Birth"])
    extracted["ID NO"] = clean_id_no(extracted["ID NO"])

    # Apply validation rules
    fields_to_validate = ['নাম', 'পিতা', 'মাতা', 'স্বামী', 'স্ত্রী']
    for field in fields_to_validate:
        if contains_english(extracted[field]):
            extracted[field] = "Not found"
    if contains_bangla(extracted["Name"]):
        extracted["Name"] = "Not found"

    return extracted



# Format OCR results for side-by-side comparison
def format_ocr_results(tesseract_results, easyocr_results):
    output = []
    for field in patterns.keys():
        tesseract_value = tesseract_results.get(field, "Not found")
        easyocr_value = easyocr_results.get(field, "Not found")
        output.append(f"tesseract -> {field}: {tesseract_value}   easy ocr -> {field}: {easyocr_value}")
    return "\n".join(output)

# Combine OCR results based on conditions
def combine_ocr_results(tesseract_results, easyocr_results):
    combined = {}
    for field in patterns.keys():
        tesseract_value = tesseract_results.get(field, "Not found")
        easyocr_value = easyocr_results.get(field, "Not found")

        # Condition 1: If both match, take Tesseract's value
        if tesseract_value == easyocr_value:
            combined[field] = tesseract_value
        # Condition 2: If both are "Not found", use "Not found"
        elif tesseract_value == "Not found" and easyocr_value == "Not found":
            combined[field] = "Not found"
        # Condition 3: If only one has data, take it
        elif tesseract_value != "Not found" and easyocr_value == "Not found":
            combined[field] = tesseract_value
        elif tesseract_value == "Not found" and easyocr_value != "Not found":
            combined[field] = easyocr_value
        # Condition 4: If they differ, take the one with more words
        else:
            tesseract_words = len(tesseract_value.split())
            easyocr_words = len(easyocr_value.split())
            combined[field] = tesseract_value if tesseract_words >= easyocr_words else easyocr_value

    # Format combined results, including all fields
    output = [f"{field}: {value}" for field, value in combined.items()]
    return "\n".join(output)

# Main function to process image with both Tesseract and EasyOCR
def main(image_path):
    # Get OCR texts
    tesseract_text = get_tesseract_ocr(image_path)
    easyocr_text = get_easyocr_text(image_path)
    #easyocr_text = ''

    # Extract fields
    tesseract_results = extract_fields(tesseract_text) if tesseract_text else {key: "Not found" for key in patterns}
    easyocr_results = extract_fields(easyocr_text) if easyocr_text else {key: "Not found" for key in patterns}

    # Format individual and combined results
    individual_results = format_ocr_results(tesseract_results, easyocr_results)
    combined_results = combine_ocr_results(tesseract_results, easyocr_results)

    # Combine outputs with a separator
    #output = f"{individual_results}\n\nCombined Results:\n{combined_results}"
    output = f"{combined_results}"

    print(output)
    # Combine outputs with a separator
    return output

# Folder containing images
folder_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/"
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(folder_path, filename)
        print(f"\nProcessing image: {filename}")
        #result = main("C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/cropped2.jpg")
        img = cv2.imread(image_path)
        rotate = dskew(img)
        preprocessed_image, original_image = preprocess_before_crop(rotate)
        result = main(preprocessed_image)
