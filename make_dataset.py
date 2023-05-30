import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt

class_n = 4
cap = cv2.VideoCapture(f'videos/{class_n}.mp4')
counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    path = 'D:\\IlyaGasanDevop\\computer_vision\\face detection\\dataset'
    counter+=1
    cv2.imwrite(f'dataset/image_class_{class_n}_{counter}.jpg', frame)
cap.release()
cv2.destroyAllWindows()