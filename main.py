from deepface import DeepFace

def verify_faces():

    backends = [
        'opencv',
        'ssd',
        'dlib',
        'mtcnn',
        'fastmtcnn',
        'retinaface',
        'mediapipe',
        'yolov8',
        'yunet',
        'centerface',
    ]

    alignment_modes = [True, False]

    obj = DeepFace.verify(
        img1_path="C:/Users/ishfaq.rahman/Desktop/6.jpg",
        img2_path="C:/Users/ishfaq.rahman/Desktop/3.jpg",
        detector_backend=backends[0],
        # align=alignment_modes[0],  # Enable alignment if desired
        # img1_path="C:/Users/ishfaq.rahman/Desktop/4.jpg",
        # img2_path="C:/Users/ishfaq.rahman/Desktop/3.jpg",
        # model_name="Facenet",
        # detector_backend="retinaface",
        # distance_metric="cosine",
        # enforce_detection=True,
    )
    print(obj)

if __name__ == "__main__":
    verify_faces()

#pip install facenet-pytorch