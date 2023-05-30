import numpy as np
import cv2
from PIL import Image
import os
import cvzone

names_txt = os.listdir('results')

class_names = ["matveu", "ilya", "sanya", "roman", "lion"]

numbers = [i for i in range(1,len(class_names)+1)]

dict_classes = dict(zip(numbers, class_names))
def finde_coords(array:list):
    return int(array[2]), int(array[3]), int(array[2])+int(array[4]), int(array[3])+int(array[5])
for name in names_txt:
    with open(f'results/{name}', 'r') as file:
        lines = file.readlines()
        for i in lines:
            array_in_line = i.split()
            array_in_line[0] = array_in_line[0].split('\\')[1]
            img = cv2.imread(f'dataset/{array_in_line[0]}')
            x1, y1, x2, y2 = finde_coords(array_in_line)

            bbox = int(x1), int(y1), int(array_in_line[4]), int(array_in_line[5])

            # confidence
            cvzone.cornerRect(img, bbox)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print(int(array_in_line[1]))
            print(dict_classes)
            cvzone.putTextRect(img, f'{x1, y1}, {dict_classes[int(array_in_line[1])]}', thickness=1, pos=(x1, y1), scale=0.8)
            cv2.imshow("Image", img)
            cv2.waitKey(300)
