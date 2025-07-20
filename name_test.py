
import binascii
from PIL import Image
from io import BytesIO
import base64
import os
import tempfile
import cv2
import numpy as np
import re
import warnings
from datetime import datetime
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CUSTOM_CACHE_DIR = "C:/TempfileForDoctr"

# det_model_dir_path = "C:/my_paddleocr_models/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer"
# rec_model_dir_path = "C:/my_paddleocr_models/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer"
# cls_model_dir_path = "C:/my_paddleocr_models/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"

# Create custom cache directory if it doesn't exist
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

# Set environment variables for Matplotlib and DocTR
os.environ["MPLCONFIGDIR"] = CUSTOM_CACHE_DIR
os.environ["DOCTR_CACHE_DIR"] = CUSTOM_CACHE_DIR

# Initialize PaddleOCR with these precise model directory paths
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='en',  # Keep lang='en' as your det and rec models are English
#     use_gpu=False,
#     show_log=False,
#     det_model_dir=det_model_dir_path,
#     rec_model_dir=rec_model_dir_path,
#     cls_model_dir=cls_model_dir_path
# )

# model load doctr
predictor = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
warnings.filterwarnings("ignore", category=UserWarning, module="paddle.utils.cpp_extension")


def get_dcotr_ocr(image_input):
    result=""
    try:
        # Ensure custom directory exists
        os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            doc = DocumentFile.from_images(image_input)
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("image_input is an empty array")
            temp_file_name = f"temp_image_{os.urandom(8).hex()}.png"
            temp_file_path = os.path.join(CUSTOM_CACHE_DIR, temp_file_name)
            try:
                cv2.imwrite(temp_file_path, image_input)
                doc = DocumentFile.from_images(temp_file_path)
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        else:
            raise TypeError("Unsupported image input type. Must be a file path (str) or a NumPy array.")

        result = predictor(doc)
        raw_text = []
        for page in result.pages:
            for block in page.blocks:
                if block.artefacts:
                    for artefact in block.artefacts:
                        raw_text.append(f"Table cell: {artefact.value}\n")
                for line in block.lines:
                    raw_text.append(" ".join([word.value for word in line.words]) + "\n")
        return "".join(raw_text).strip()

    except Exception:
        return result


# def get_paddle_ocr(image):
#     all_text = ""
#     try:
#         results = ocr.predict(image)
#         if results and isinstance(results, list):
#             for page in results:
#                 if page and isinstance(page, dict):
#                     rec_texts = page.get('rec_texts', [])
#                     for text in rec_texts:
#                         if text and text.strip():
#                             all_text += text.strip() + "\n"
#         return all_text.strip()
#     except Exception:
#         return all_text


def resize_image_in_memory(input_data):
    if isinstance(input_data, str):
        img = Image.open(input_data).convert('RGB')
    elif isinstance(input_data, BytesIO):
        img = Image.open(input_data).convert('RGB')
    else:
        raise ValueError("Input must be a file path or BytesIO object.")

    original_width, original_height = img.size
    longest_side = max(original_width, original_height)

    if longest_side <= 2000:
        return img
    elif longest_side <= 4000:
        scale_factor = 0.8
    elif longest_side <= 100000:
        scale_factor = 0.6
    else:
        raise ValueError("Image dimensions exceed 100000 pixels.")

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img


def extract_name_from_paddleOcr(raw_data: str) -> str | None:
   lines = [line.strip() for line in raw_data.strip().split('\n')]
   name_line_index = -1

   for i, line in enumerate(lines):
       if re.search(r'^Name[:：.\s-]*', line, re.IGNORECASE):
           name_line_index = i
           break

   if name_line_index == -1:
       return ""

   name_line_content = lines[name_line_index]
   match = re.search(r'Name[:：.\s-]*(.+)', name_line_content, re.IGNORECASE)

   if match:
       potential_name = match.group(1).strip()
       is_valid_name = any(c.isalpha() for c in potential_name)

       if is_valid_name and ':' not in potential_name.split(' ')[0]:
           return potential_name

   if name_line_index + 1 < len(lines):
       next_line = lines[name_line_index + 1]

       if ':' in next_line:
           if name_line_index > 0:
               previous_line = lines[name_line_index - 1]
               if previous_line:
                   return previous_line
           return ""
       else:
           if any(c.isalpha() for c in next_line):
               return next_line
           elif name_line_index > 0:
               previous_line = lines[name_line_index - 1]
               if previous_line:
                   return previous_line

   return ""



def process_raw_data(text):
    text_without_dates = text
    extracted = {}
    name_paddle = ''
    dob_paddle = ''
    nid_paddle = ''
    # Pattern for DDMonYYYY and DD Mon YYYY formats with optional multiple separators and newlines
    pattern_name = re.compile(r'Name[:：]*\s*([^\n:]+)', re.IGNORECASE | re.MULTILINE)
    pattern_date = re.compile(r'\d{2}[\s.\n-]*?[A-Za-z]{3}[\s.\n-]*?\d{4}')

    # Find all date matches
    matches_name = pattern_name.findall(text)
    matches_date = pattern_date.findall(text)

    # if matches_name:
    #     try:
    #         name_paddle = re.sub(r"[^A-Za-z\s\.\-\u0980-\u09FF]", "", matches_name[0]).strip()
    #
    #     except Exception:
    #         name_paddle = ""
    #
    # else:
    #     name_paddle = ""
    try:
        name_paddle = extract_name_from_paddleOcr(text)
        name_paddle = re.sub(r'\.(\w)', r'. \1', name_paddle)
    except Exception:
        name_paddle = ""

    if matches_date:
        try:
            # DOB = matches_date[0]
            if len(matches_date) > 1:
                dob_paddle = re.sub(r'[\s.\n-]+', '', matches_date[1])
            else:
                dob_paddle = re.sub(r'[\s.\n-]+', '', matches_date[0])

            date_obj = datetime.strptime(dob_paddle, "%d%b%Y")
            # Convert to YYYY-MM-DD
            dob_paddle = date_obj.strftime("%Y-%m-%d")
            # Remove dates from text
            text_without_dates = pattern_date.sub('', text)

        except Exception:
            dob_paddle = None

    else:
        dob_paddle = None

    # Pattern for 10, 13, 17 digit numbers with optional spaces, dashes, dots, newlines
    pattern_num = re.compile(r'((?:\d[\s.\n-]*){17}|(?:\d[\s.\n-]*){13}|(?:\d[\s.\n-]*){10})')
    matches_num = pattern_num.findall(text_without_dates)

    if matches_num:
        try:
            nid_paddle = re.sub(r'[\s.\n-]+', '', matches_num[0])

        except Exception:
            nid_paddle = None

    else:
        nid_paddle = None

    extracted['Name'] = name_paddle
    extracted['DateOfBirth'] = dob_paddle
    extracted['IDNO'] = nid_paddle

    return extracted





def process_image_raw(image_path):
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise ValueError(f"Failed to load image at {image_path}")

    # for paddle process
    paddle_text1 = get_dcotr_ocr(img_cv2)
    return paddle_text1

def process_image(image_path):
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise ValueError(f"Failed to load image at {image_path}")

    # for paddle process
    paddle_text1 = get_dcotr_ocr(img_cv2)
    # print(paddle_text1)
    paddle_result = process_raw_data(paddle_text1)
    # print('paddle prccess data: ', paddle_result)

    # for dcotr process
    #doctr_text1 = get_dcotr_ocr(image_path)
    doctr_text1 = ""
    # print(doctr_text1)
    docrt_result = process_raw_data(doctr_text1)
    # print('doctr prccess data: ', docrt_result)

    final_name = {}
    if paddle_result['Name'].replace(" ", "") == docrt_result['Name'].replace(" ", ""):

        if len(paddle_result['Name']) > len(docrt_result['Name']):
            final_name['Name'] = paddle_result['Name']
        else:
            final_name['Name'] = docrt_result['Name']
    else:
        if len(paddle_result['Name']) == len(docrt_result['Name']):
            final_name['Name'] = paddle_result['Name']
        elif len(paddle_result['Name']) > len(docrt_result['Name']):
            final_name['Name'] = paddle_result['Name']
        else:
            final_name['Name'] = docrt_result['Name']

    final_name['DateOfBirth'] = paddle_result['DateOfBirth'] or docrt_result['DateOfBirth'] or ""
    final_name['IDNO'] = paddle_result['IDNO'] or docrt_result['IDNO'] or ""
    final_name['পিতা'] = ""
    final_name['মাতা'] = ""
    final_name['স্বামী'] = ""
    final_name['স্ত্রী'] = ""
    final_name['নাম'] = ""

    return final_name


app = FastAPI()
class ImageData(BaseModel):
    image_base64: str


@app.post("/extract-fields/")
async def extract_fields(data: ImageData):
    try:
        # Decode base64 to image
        image_bytes = base64.b64decode(data.image_base64)

        # Convert bytes to PIL Image for resizing
        #img = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Resize the image using your resize function
        # resized_img = resize_image_in_memory(BytesIO(image_bytes))

        # Save resized image to temporary file
        with tempfile.NamedTemporaryFile(dir="C:/TempfileForDoctr", suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        # Process the image with the custom processing function
        result = process_image(temp_file_path)

        # Delete the temporary file
        os.unlink(temp_file_path)

        return result

    except binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-raw/")
async def extract_fields(data: ImageData):
    try:
        # Decode base64 to image
        image_bytes = base64.b64decode(data.image_base64)

        # Convert bytes to PIL Image for resizing
        #img = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Resize the image using your resize function
        # resized_img = resize_image_in_memory(BytesIO(image_bytes))

        # Save resized image to temporary file
        with tempfile.NamedTemporaryFile(dir="C:/TempfileForDoctr", suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        # Process the image with the custom processing function
        result = process_image_raw(temp_file_path)

        # Delete the temporary file
        os.unlink(temp_file_path)

        return result

    except binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/7-Jul-2025/3.png"
# result = process_image(image_path)
# print(result)
