pip install insightface
pip install onnxruntime-gpu
# onnx                         1.18.0
# onnxruntime-gpu              1.19.2
pip install numpy opencv-python

import insightface
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace with ArcFace model
app = FaceAnalysis(name='arcface_r100_v1', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  # Optimized for full VRAM

# Load two images
img1 = cv2.imread(r'C:\Users\Administrator\Desktop\Images\1.jpg')
img2 = cv2.imread(r'C:\Users\Administrator\Desktop\Images\1.jpg')

# Detect and extract embeddings
faces1 = app.get(img1)
faces2 = app.get(img2)

if len(faces1) == 0 or len(faces2) == 0:
    print("No faces detected in one or both images.")
    exit()

# Get embeddings
embedding1 = faces1[0].normed_embedding
embedding2 = faces2[0].normed_embedding

# Compute cosine similarity
similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
print(f"Raw similarity score: {similarity:.4f}")



pip install deepface
pip install tensorflow-gpu
pip install opencv-python


from deepface import DeepFace
import cv2

img1 = cv2.imread(r'C:\Users\Administrator\Desktop\Images\1.jpg')
img2 = cv2.imread(r'C:\Users\Administrator\Desktop\Images\1.jpg')

result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name='ArcFace', distance_metric='cosine')

print(result)


