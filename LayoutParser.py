import layoutparser as lp
import cv2
import pytesseract
from PIL import Image

# Load the image
image_path = "C:/Users/ishfaq.rahman/Desktop/5.jpg"  # Change this to your NID image path
image = cv2.imread(image_path)

# Convert BGR (OpenCV) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load layout detection model
model = lp.PaddleDetectionLayoutModel(
    model_path='lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config',
    label_map={0: "Text"},
    enforce_cpu=True  # set False if you have GPU + paddlepaddle-gpu installed
)

# Detect layout blocks
layout = model.detect(image_rgb)

# Optional: Draw layout boxes
drawn_image = lp.draw_box(image_rgb, layout, box_width=3)
cv2.imshow("Layout", cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)

# Iterate and extract text from each block
all_text = ""
for block in layout:
    x_1, y_1, x_2, y_2 = map(int, block.coordinates)
    cropped_region = image_rgb[y_1:y_2, x_1:x_2]

    # Convert to PIL image
    pil_img = Image.fromarray(cropped_region)

    # Apply OCR
    text = pytesseract.image_to_string(pil_img, lang='eng+ben')  # English + Bengali
    print(f"Detected Text Block:\n{text}\n{'-'*50}")
    all_text += text + "\n"

#

