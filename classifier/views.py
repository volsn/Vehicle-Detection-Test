from django.shortcuts import render
from django.http.response import StreamingHttpResponse, HttpResponse
from classifier.models import Camera, Shot

from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.auth.decorators import login_required

import numpy as np
import threading
import cv2
import os
import requests
import time
from PIL import Image
import pickle
import random
import string


class VehicleDetector:
    def __init__(self, config):
        self.min_confidence = config['min_confidence']
        self.threshold = config['threshold']
        self.classes = config['labels']
        self.min_sizes = config['min_sizes']

    def fit(self, yolo_path):
        labelsPath = os.path.sep.join([yolo_path, "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")

        weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
        configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in \
            self.net.getUnconnectedOutLayers()]

    def predict(self, image):
        return self._get_objects(image)

    def predict_proba(self, image):
        return self._get_objects(image, add_proba=True)

    def _get_objects(self, image, add_proba=False):

        image = image.copy()
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.min_confidence:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, \
                self.min_confidence, self.threshold)

        output = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                if self.LABELS[classIDs[i]] in self.classes:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    if self.min_sizes is not None:
                        if w < self.min_sizes['width'] \
                                or h < self.min_sizes['height']:
                            continue

                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if x+w > W:
                        w = W - x
                    if y+h > H:
                        h = H - y

                    if add_proba:
                        output.append(((y, h, x, w), self.LABELS[classIDs[i]], \
                                confidences[i]))
                    else:
                        output.append(((y, h, x, w), self.LABELS[classIDs[i]]))

        return output

def classify_sklearn(model, image):
    image = cv2.resize(image, (64, 64))
    W, H, C = image.shape
    image = image.reshape((W * H * C))
    image = np.array([image]).astype(np.float32)
    output = model.predict(image)

    if output[0] == 1:
        return 1
    else:
        return 0


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def save_image(camera, image, roi, label):
    filename = 'temp/{}.jpg'.format(get_random_string(10))
    filename_roi = 'temp/{}.jpg'.format(get_random_string(10))
    cv2.imwrite(filename, image)
    cv2.imwrite(filename_roi, roi)
    shot = Shot(camera=camera, type=label)
    shot.image.save(
        shot.generate_name() + '.jpg',
        open(filename, 'rb')
    )
    shot.car.save(
        shot.generate_name() + '.jpg',
        open(filename_roi, 'rb')
    )
    shot.save()

# Create your views here.

def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def read_camera(camera):

    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)

    detector = \
        VehicleDetector({
            "labels": ["car", "truck", "bus"],
            "min_confidence": 0.5,
            "threshold": 0.3,
            "min_sizes": {
                "width": 50,
                "height": 50
        }})
    detector.fit('yolo-coco')

    mask = cv2.imread(os.path.join(settings.MEDIA_ROOT, str(camera.mask)))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(camera.ip_adress)
    thread = threading.currentThread()
    print(camera.seconds)

    count = 0
    if cap.isOpened():
        while True:

            if not getattr(thread, "do_run", True):
                return

            count += 1
            print('foo', count)
            ret, image = cap.read()

            try:
                orig = image.copy()
            except:
                cap = cv2.VideoCapture(camera.ip_adress)
                continue

            if count % camera.seconds == 0:
                cv2.imwrite('test_{}.png'.format(camera.pk), image)
                print('foo-count', count)

                try:
                    image = cv2.bitwise_and(image, image, mask=mask)
                except:
                    cap = cv2.VideoCapture(camera.ip_adress)
                    continue

                try:
                    boxes = detector.predict(image)
                except:
                    cap = cv2.VideoCapture(camera.ip_adress)
                    continue

                for box, _ in boxes:
                    (y, h, x, w) = box
                    roi = orig[y: y+h, x: x+w]
                    cv2.imwrite(get_random_string(10) + '.png', roi)

                    print('bar')
                    try:
                        label = classify_sklearn(model, roi)
                    except:
                        cap = cv2.VideoCapture(camera.ip_adress)
                        continue

                    if label == 0:
                        requests.get(camera.open_link)
                    print('             saved')
                    save_image(camera, orig, orig[y: y+h, x: x+w], label)


threads = {}

@login_required
def start(request, pk):
    camera = Camera.objects.get(pk=pk)

    if camera.active:
        return HttpResponse('Ошибка! Камера уже запущена')

    t = threading.Thread(target=read_camera, args=(camera,))
    threads[camera.pk] = t
    t.start()

    camera.active = True
    camera.save()

    return HttpResponse('Была запущена камера: {}'.format(camera.adress))

@login_required
def start_all(request):
    cameras = Camera.objects.all()
    for camera in cameras:
        if not camera.active:
            t = threading.Thread(target=read_camera, args=(camera,))
            threads[camera.pk] = t
            t.start()
            camera.active = True
            camera.save()

    return HttpResponse('Запущенны все камеры')

@login_required
def stop(request, pk):
    camera = Camera.objects.get(pk=pk)
    if not camera.active:
        return HttpResponse('Ошибка! Камера уже выключена')

    threads[pk].do_run = False
    threads[pk].join()

    camera.active=False
    camera.save()

    return HttpResponse('Камера {} была отключена'\
                .format(Camera.objects.get(pk=pk).adress))

@login_required
def stop_all(request):
    for pk, thread in threads.items():
        camera = Camera.objects.get(pk=pk)
        if camera.active:

            thread.do_run = False
            thread.join()

            camera.active = False
            camera.save()

    return HttpResponse('Все камеры были выключены')

@login_required
def partial_train(request):

    # Loading model
    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)

    # Loading training instances
    X = []
    y = []
    shots = Shot.objects.filter(wrong_label=True)
    for shot in shots:
        image = cv2.imread(os.path.join(settings.MEDIA_ROOT, str(shot.car)))
        image = cv2.resize(image, (64, 64))
        W, H, C = image.shape
        image = image.reshape((W * H * C))

        X.append(image)
        y.append(shot.type)

        shot.wrong_label = False
        shot.save()

    X = np.array(X).astype(np.float32)
    y = np.array(y)

    if len(X) == 0:
        return HttpResponse('Ошибка! Не было выбрано ни одного фото')

    model.partial_fit(X, y)

    with open('model.pickle', 'wb') as file:
        pickle.dump(model, file)

    return HttpResponse('Модель дообучена!')
