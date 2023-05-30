import cv2 as cv
import numpy as np
from PIL import Image
import os
import asyncio
import time
import cvzone

ames_txt = os.listdir('results')

class_names = ["matveu", "ilya", "sanya", "roman", "lion"]

numbers = [i for i in range(1,len(class_names)+1)]

dict_classes = dict(zip(numbers, class_names))


cascadePath = "haarcascade_frontalface_default.xml"
profilePath = "haarcascade_profileface.xml"

faceCascade = cv.CascadeClassifier(cascadePath)
faceProf = cv.CascadeClassifier(profilePath)

# Для распознавания используем локальные бинарные шаблоны
recognizer = cv.face.LBPHFaceRecognizer_create(1,8,8,8,123)
recognizerProfile = cv.face.LBPHFaceRecognizer_create(1,8,8,8,123)


async def show(image, images, labels, subject_number, faces2):
    for (x, y, w, h) in faces2:
        images.append(image[y: y + h, x: x + w])
        labels.append(subject_number)
        # В окне показываем изображение
        cv.imshow("prof", image[y: y + h, x: x + w])
        cv.waitKey(600)


def get_images(path):
    # Ищем все фотографии и записываем их в image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path) ]
    print(image_paths)
    images = []
    labels = []

    for image_path in image_paths:
        # Переводим изображение в черно-белый формат и приводим его к формату массива
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        # Из каждого имени файла извлекаем номер человека, изображенного на фото
        subject_number = int(image_path.split('_')[2])

        # Определяем области где есть лица
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(300, 300))
        # image – исходное изображение
        # scaleFactor – определяет то, на сколько будет увеличиваться скользящее окно поиска на каждой итерации. 1.1 означает на 10%, 1.05 на 5% и т.д. Чем больше это значение, тем быстрее работает алгоритм.
        # minNeighbors — Чем больше это значение, тем более параноидальным будет поиск и тем чаще он будет пропускать реальные лица, считая, что это ложное срабатывание. Оптимальное значение 3-6.
        # minSize – минимальный размер лица на фото. 30 на 30 обычно вполне достаточно.

        # Если лицо нашлось добавляем его в список images, а соответствующий ему номер в список labels




        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(subject_number)
            # # В окне показываем изображение
            # cv.imshow("front", image[y: y + h, x: x + w])
            # cv.waitKey(1)

    return images, labels



path = './dataset'
# Получаем лица и соответствующие им номера
images, labels = get_images(path)
cv.destroyAllWindows()

# Обучаем программу распознавать лица
recognizer.train(images, np.array(labels))

cap = cv.VideoCapture(0)
while True:
    ret, img = cap.read()

    # cv.imshow("camera", img)
    # cv.waitKey(100)

    # Ищем лица на фотографиях
    imgarray = np.array(img)
    gray = Image.fromarray(imgarray).convert('L')
    image = np.array(gray, 'uint8')
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    for (x, y, w, h) in faces:
        print(x, y, w, h)

        number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])
        bbox = int(x), int(y), int(w), int(h)
        cvzone.cornerRect(img, bbox)
        print()
        try:
            cvzone.putTextRect(img, f'{x, y}, {dict_classes[int(number_predicted)]}', thickness=1, pos=(x, y),
                               scale=0.8)
        except:
            continue
    cv.imshow("Image", img)
    cv.waitKey(1)

    # for (x, y, w, h) in faces:
    #     # Если лица найдены, пытаемся распознать их
    #     # Функция  recognizer.predict в случае успешного распознавания возвращает номер и параметр confidence,
    #     # этот параметр указывает на уверенность алгоритма, что это именно тот человек, чем он меньше, тем больше уверенность
    #     number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])
    #
    #     # Извлекаем настоящий номер человека на фото и сравниваем с тем, что выдал алгоритм
    #     number_actual = int(image_path.split('_')[3])
    #
    #     if number_actual == number_predicted:
    #         print("{} is Correctly Recognized with confidence {}".format(number_actual, conf))
    #     else:
    #         print("{} is Incorrect Recognized as {}".format(number_actual, number_predicted), 'parasms', x, y, w, h,
    #               image_path)
    #
    #     with open(f"results/res_class{number_predicted}.txt", "a") as file:
    #         file.write(f"{image_path} {number_predicted} {x} {y} {w} {h} \n")
    #     cv.imshow("Recognizing Face", image[y: y + h, x: x + w])
    #     cv.waitKey(1)
