from django.shortcuts import render
from django.http.response import StreamingHttpResponse, HttpResponse
from classifier.models import Camera, Shot

from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.base import ContentFile

import numpy as np
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
        return 'civilian'
    else:
        return 'ambulance'


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def save_image(camera, image):
    filename = 'temp/{}.jpg'.format(get_random_string(10))
    cv2.imwrite(filename, image)
    shot = Shot(camera=camera)
    shot.image.save(
        shot.generate_name() + '.jpg',
        open(filename, 'rb')
    )
    shot.save()

# Create your views here.

def stream(cap, camera):

    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)

    detector = \
        VehicleDetector({
            "labels": ["car", "truck", "bus"],
            "min_confidence": 0.3,
            "threshold": 0.3,
            "min_sizes": {
                "width": 50,
                "height": 50
        }})
    detector.fit('yolo-coco')

    while True:

        try:
            ret, image = cap.read() # Reading image from videocap
            original = image.copy()
        except:
            continue

        try:
            boxes = detector.predict(image) # Detecting boxes
        except:
            continue

        for box, _ in boxes:
            (y, h, x, w) = box
            roi = image[y: y+h, x: x+w]

            try:
                label = classify_sklearn(model, roi) # Classifying the vehicle
            except:
                continue

            if label == 'ambulance':
                color = (0, 255, 0)

                if camera.active:
                    try:
                        requests.get(camera.open_link)
                    except:
                        pass
                    save_image(camera, original)
            else:
                color = (0, 0, 255)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


        ret, jpeg = cv2.imencode('.jpg', image)
        jpeg = jpeg.tobytes()

        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')
        time.sleep(1)

def stream_video_view(request, pk):
    camera = Camera.objects.get(pk=pk)
    ip_adress = camera.ip_adress
    cap = cv2.VideoCapture(ip_adress)

    response = StreamingHttpResponse(stream(cap, camera), \
                content_type="multipart/x-mixed-replace;boundary=frame")
    return response


def classify_view(request, sec):
    cameras = Camera.objects.filter(active=True)

    while True:
        for camera in cameras:
            cap = cv2.VideoCapture(camera.ip_adress)
            ret, image = cap.read()
            boxes = detector.predict(image)
            for box, _ in boxes:
                (y, h, x, w) = box
                roi = image[y: y+h, x: x+w]
                label = classify_sklearn(model, roi)
                if label == 'ambulance':

                    # Opening the gates
                    #requests.get(camera.open_link)

                    # Saving camera shot to SQL
                    save_image(camera, image)

        time.sleep(sec)

    return HttpResponse('foo')
