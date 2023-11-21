from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
import psutil
import cv2
from ultralytics import YOLO
import tensorflow as tf
import numpy as np


license_plate_detector = YOLO("./last.pt")
upper_digit_recognizer = tf.keras.models.load_model("upper_digit_recognizer.h5")
lower_digit_recognizer = tf.keras.models.load_model("lower_digit_recognizer.h5")

class Window(QMainWindow):
    def __init__(self, license_plate_detector = license_plate_detector, upper_digit_recognizer = upper_digit_recognizer, 
                 lower_digit_recognizer = lower_digit_recognizer):
        super().__init__()
        uic.loadUi("./GUI.ui", self)
        self.license_plate_detector = license_plate_detector
        self.upper_digit_recognizer = upper_digit_recognizer
        self.lower_digit_recognizer = lower_digit_recognizer

        self.setWindowTitle("Phone Use Detection")
        self.label_9.setAlignment(Qt.AlignCenter)
        self.label_14.setAlignment(Qt.AlignCenter)
        self.label.setAlignment(Qt.AlignCenter)

        self.pushButton.clicked.connect(self.execute)  # function execute
        self.pushButton_2.clicked.connect(self.choose_image)
        self.image = None

        self.showTime()

    def execute(self):
        patch = self.detectLicensePlate(self.image)
        if not isinstance(patch, str):
            upperRes, lowerRes = self.recognizeDigits(patch)
            if lowerRes[-1] == "e": lowerRes = lowerRes[:-1]
            self.label_9.setText(upperRes + " " + lowerRes)
        else:
            self.label_9.setText("Not Detected")

    def update_image(self):
        self.image = cv2.imread(self.img_dir)
        if self.image is not None:
            image_display = QImage(self.image, self.image.shape[1], self.image.shape[0], QImage.Format_BGR888)
            self.label.setPixmap(QPixmap.fromImage(image_display))
            self.label_9.setText("Not executed")
        else:
            self.label.setText("Image format is not supported")

    def choose_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
            options=options,
        )
        if file_name:
            self.img_dir = file_name
            self.update_image()
            
    def showTime(self):
        datetime = QDateTime.currentDateTime()
        text = datetime.toString("dddd MMMM yyyy")
        self.label_14.setText(text)
        self.lcdNumber.display(datetime.toString("hh:mm"))
        cpu_percent = psutil.cpu_percent()
        self.label_17.setText(str(cpu_percent) + "%")
        self.label_18.setText(str(15) + "%")
        QTimer.singleShot(1000, self.showTime)


    def detectLicensePlate(self, image):
        pred = self.license_plate_detector(image, verbose=False)
        if len(pred[0].boxes.data.cpu().numpy()) > 0:
            x1, y1, x2, y2, _, _ = pred[0].boxes.data.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            patch = image[y1:y2, x1:x2, :].copy()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            return patch
        return "Error"
    
    def recognizeDigits(self, license_plate):
        license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
        license_plate = license_plate/255.
        height, width = license_plate.shape
        upperPart = license_plate[:height//2, :]
        upperPart = cv2.resize(upperPart, (95, 42))
        upperPart = upperPart.T
        upperPart = upperPart[None, ..., None]
        upperRes = upper_digit_recognizer(upperPart).numpy().decode()
        
        lowerPart = license_plate[height//2:height, :]
        lowerPart = cv2.resize(lowerPart, (96, 45))
        lowerPart = lowerPart.T
        lowerPart = lowerPart[None, ..., None]
        lowerRes = lower_digit_recognizer(lowerPart).numpy().decode()
        
        return upperRes, lowerRes


if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.show()
    app.exec_()