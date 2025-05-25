from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import sys

# Load the Moondream2 model and tokenizer
model_id = "vikhyatk/moondream2"
revision = "2025-01-09"
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        # Uncomment for CUDA if available:
        # device_map={"": "cuda"},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Load and prepare the image
image_path = "C:/Users/ishfaq.rahman/Desktop/6.jpg"  # Replace with your image path
try:
    image = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}. Please check the path.")
    sys.exit(1)

# Encode the image
try:
    encoded_image = model.encode_image(image)
except Exception as e:
    print(f"Error encoding image: {e}")
    sys.exit(1)

# Generate a caption
try:
    short_caption = model.caption(image, length="short")["caption"]
    print("Short Caption:", short_caption)
except Exception as e:
    print(f"Error generating caption: {e}")

# Ask a question about the image
query = "What is the main object in the image?"
try:
    answer = model.query(image, query)["answer"]
    print(f"Query: {query}")
    print(f"Answer: {answer}")
except Exception as e:
    print(f"Error answering query: {e}")

# Perform OCR
ocr_query = "Transcribe the text in natural reading order"
try:
    ocr_result = model.query(image, ocr_query)["answer"]
    print(f"OCR Result: {ocr_result}")
except Exception as e:
    print(f"Error performing OCR: {e}")