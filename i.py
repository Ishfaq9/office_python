import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.button import Button

import numpy as np
import cv2
from jnius import autoclass

# Android classes for gallery refresh
Environment = autoclass('android.os.Environment')
MediaScannerConnection = autoclass('android.media.MediaScannerConnection')
PythonActivity = autoclass('org.kivy.android.PythonActivity')


class AutoCaptureCamera(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.cam = Camera(play=True, resolution=(640, 480))
        self.cam.size_hint = (1, 0.85)
        self.cam.allow_stretch = True
        self.cam.keep_ratio = True

        self.status = Label(text="Looking for card...", size_hint=(1, 0.15))

        self.layout.add_widget(self.cam)
        self.layout.add_widget(self.status)

        self.detection_start_time = None
        self.hold_time_sec = 2  # seconds to hold detection before capture
        self.captured = False

        Clock.schedule_interval(self.detect_card, 1.0 / 10.0)  # 10 times per second

        return self.layout

    def detect_card(self, dt):
        if not self.cam.texture:
            return

        texture = self.cam.texture
        size = texture.size
        pixels = texture.pixels

        img = np.frombuffer(pixels, np.uint8)
        img = img.reshape(size[1], size[0], 4)  # height, width, RGBA
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img_bgr = cv2.flip(img_bgr, 0)  # Flip vertically

        if self.is_card_detected(img_bgr):
            now = Clock.get_time()
            if self.detection_start_time is None:
                self.detection_start_time = now
                self.status.text = "Card detected, hold steady..."
            elif now - self.detection_start_time >= self.hold_time_sec and not self.captured:
                self.captured = True
                self.status.text = "Hold time passed! Capturing image..."
                self.save_image_to_gallery(img_bgr)
                self.show_popup("Card detected successfully!")
        else:
            self.status.text = "Looking for card..."
            self.detection_start_time = None
            self.captured = False

    def is_card_detected(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 75, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            area = cv2.contourArea(cnt)

            if len(approx) == 4 and area > 20000:
                return True
        return False

    def save_image_to_gallery(self, img_bgr):
        # Rotate image 90 degrees clockwise to fix default rotation
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

        # Save to public Pictures/CardDetectorApp folder
        pictures_dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).getAbsolutePath()
        app_folder = os.path.join(pictures_dir, "CardDetectorApp")
        if not os.path.exists(app_folder):
            os.makedirs(app_folder)

        filename = os.path.join(app_folder, "captured_card.jpg")
        cv2.imwrite(filename, img_bgr)
        print(f"✅ Image saved to: {filename}")

        # Refresh gallery to show image in Photos app
        self.refresh_gallery(filename)

    def refresh_gallery(self, filepath):
        activity = PythonActivity.mActivity
        MediaScannerConnection.scanFile(activity, [filepath], None, None)
        print("Gallery refreshed.")

    def show_popup(self, message):
        popup_content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        popup_content.add_widget(Label(text=message))
        close_button = Button(text="OK", size_hint=(1, 0.3))
        popup_content.add_widget(close_button)

        popup = Popup(title="Notification", content=popup_content,
                      size_hint=(0.7, 0.4), auto_dismiss=False)

        close_button.bind(on_press=popup.dismiss)
        popup.open()

    def on_stop(self):
        self.cam.play = False


if __name__ == '__main__':
    AutoCaptureCamera().run()
