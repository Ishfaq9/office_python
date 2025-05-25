from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window

from kivy.graphics.texture import Texture
from kivy.clock import Clock

import numpy as np
import cv2
from PIL import Image as PILImage
import os
from datetime import datetime

class CardDetector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Camera widget
        self.camera = Camera(play=True)
        self.camera.resolution = (1280, 720)  # May still be low on Android
        self.add_widget(self.camera)

        # Button to capture
        self.capture_btn = Button(text="Capture Image", size_hint=(1, 0.1))
        self.capture_btn.bind(on_press=self.capture_image)
        self.add_widget(self.capture_btn)

        # Status message
        self.status_label = Label(text="", size_hint=(1, 0.1))
        self.add_widget(self.status_label)

    def capture_image(self, instance):
        texture = self.camera.texture
        if texture:
            # Get image from camera
            size = texture.size
            pixels = texture.pixels
            pil_img = PILImage.frombytes(mode='RGBA', size=size, data=pixels)
            pil_img = pil_img.transpose(PILImage.FLIP_TOP_BOTTOM)  # Fix orientation

            # Convert to OpenCV BGR image
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)

            # Detect card-like shape (rectangle with 4 corners)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            card_found = False
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                area = cv2.contourArea(cnt)
                if len(approx) == 4 and area > 10000:
                    card_found = True
                    cv2.drawContours(cv_img, [approx], 0, (0, 255, 0), 4)
                    break

            # Message display
            if card_found:
                self.status_label.text = "✅ Card detected successfully!"
            else:
                self.status_label.text = "⚠️ No card detected."

            # Save the image to gallery
            save_dir = "/storage/emulated/0/DCIM/Camera"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"card_detected_{timestamp}.jpg")
            try:
                cv2.imwrite(save_path, cv_img)
                self.status_label.text += f"\n📷 Image saved: {save_path}"
            except Exception as e:
                self.status_label.text += f"\n❌ Failed to save image: {e}"

class CardApp(App):
    def build(self):
        Window.clearcolor = (0, 0, 0, 1)
        return CardDetector()

if __name__ == '__main__':
    CardApp().run()
