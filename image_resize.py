from PIL import Image
import time
import base64
from io import BytesIO

def resize_image_in_memory(input_path):
    img = Image.open(input_path).convert('RGB')
    original_width, original_height = img.size
    longest_side = max(original_width, original_height)

    if longest_side <= 2000:
        return img
    elif longest_side <= 4000:
        scale_factor = 0.5
    elif longest_side <= 100000:
        scale_factor = 0.3
    else:
        raise ValueError("Image dimensions exceed 100000 pixels.")

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img



# Main

image_path = "C:/Users/ishfaq.rahman/Desktop/NID Images/New Images/NID_3.png"
resized_img = resize_image_in_memory(image_path)

if resized_img:
    buffered = BytesIO()
    resized_img.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print(encoded_string)
