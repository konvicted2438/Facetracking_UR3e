import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFrame
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
import URBasic
import math
import numpy as np
import sys
import cv2
import time
import imutils
from imutils.video import VideoStream
import math3d as m3d
import mediapipe as mp
from mss import mss
# Import the face tracking module
import CAMTracking.FaceTracking1 as FT
bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}
sct = mss()
# Thread for handling video capture and processing
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.features_to_track = {'lips': False, 'nose': False, 'left_eye': False, 'right_eye': False, 'left_eyebrow': False, 'right_eyebrow': False, 'face_shape': True}

    def update_tracking_features(self, feature, status):
        if status:
            if feature not in self.features_to_track:
                self.features_to_track[feature] = True
        else:
            self.features_to_track.pop(feature, None)
            

    def run(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        face_tracker = FT.FaceTracking()
        
        while True:
            ret, frame = cap.read()
            #sct_img = sct.grab(bounding_box)
            if not ret:
                break
            #cv2.imshow(np.array(sct_img))
            #if (cv2.waitKey(1) & 0xFF) == ord('q'):
            #    cv2.destroyAllWindows()
            #    break
            active_features = list(self.features_to_track.keys())
            processed_frame, _ = face_tracker.track_specific_features(frame, active_features)
            self.change_pixmap_signal.emit(processed_frame)

        cap.release()

# Main GUI window
class TrackingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Tracking GUI")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def init_ui(self):
        main_layout = QHBoxLayout()

        self.video_frame = QLabel("Video Display")
        self.video_frame.setFixedSize(640, 480)
        self.video_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.video_frame.setAlignment(Qt.AlignCenter)
        
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)

        # Add buttons for each feature
        self.feature_buttons = {}
        for feature in ['lips', 'nose', 'left_eye', 'right_eye','left_eyebrow', 'right_eyebrow', 'face_shape']:
            button = QPushButton(f"Track {feature.capitalize()}")
            button.setCheckable(True)
            button.toggled.connect(lambda checked, f=feature: self.toggle_feature(checked, f))
            control_layout.addWidget(button)
            self.feature_buttons[feature] = button

        main_layout.addWidget(self.video_frame)
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)

    def toggle_feature(self, checked, feature):
        self.thread.update_tracking_features(feature, checked)

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_frame.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.thread.terminate()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = TrackingGUI()
    gui.show()
    sys.exit(app.exec_())
