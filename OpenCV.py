from paddleocr import PaddleOCR

ocr = PaddleOCR(
    device="gpu:0",
    lang="en",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def get_paddle_ocr(image):
    results = ocr.ocr(image)
    all_text = ""
    if results and isinstance(results, list):
        for page in results:
            if page and isinstance(page, dict):
                rec_texts = page.get('rec_texts', [])
                for text in rec_texts:
                    if text and text.strip():
                        all_text += text.strip() + "\n"
    return all_text.strip()


from paddleocr import PaddleOCR

# Initialize PaddleOCR with PP-OCRv5 from custom directory
ocr = PaddleOCR(
    device="gpu:0",
    lang="en",
    text_detection_model_dir="C:/paddle_model/PP-OCRv5_server_det",
    text_recognition_model_dir="C:/paddle_model/PP-OCRv5_server_rec",
    textline_orientation_model_dir=None,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    #text_det_limit_side_len=7000
)

def get_paddle_ocr(image):
    results = ocr.predict(image)
    all_text = ""
    if results and isinstance(results, list):
        for page in results:
            if page and isinstance(page, dict):
                rec_texts = page.get('rec_texts', [])
                for text in rec_texts:
                    if text and text.strip():
                        all_text += text.strip() + "\n"
    return all_text.strip()

image_path = "C:/Users/Administrator/Desktop/NID Images/New Images/NID_3.png"
final_results = get_paddle_ocr(image_path)
print(final_results)


#predict