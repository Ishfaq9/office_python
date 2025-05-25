# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
#
# model = ocr_predictor(pretrained=True, pretrained_backbone=False, detect_language=True)
#
# # images
# doc = DocumentFile.from_images("C:/Users/ishfaq.rahman/Desktop/5.jpg")
# # Analyze
# result = model(doc)
#
# print(result)

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_images("C:/Users/ishfaq.rahman/Desktop/5.jpg")
result = model(doc)
print(result.render())
