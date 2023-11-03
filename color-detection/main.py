import cv2
from PIL import Image
import numpy as np


lower_limit = np.array([40, 100, 50], dtype=np.uint8)
upper_limit = np.array([80, 255, 255], dtype=np.uint8)

image = cv2.imread("green-lemon.jpg")
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsvImage, lower_limit, upper_limit)

while True:
    cv2.imshow("Image", mask)
    if cv2.waitKey(33) == ord("q"):
        break

cv2.destroyAllWindows()