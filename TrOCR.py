from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

image = Image.open("C:/Users/ishfaq.rahman/Desktop/5.jpg").convert("RGB")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
