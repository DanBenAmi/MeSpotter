import cv2
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dlib
import sys


class face_detectors:
    def __init__(self,
                 haar=cv2.CascadeClassifier(r'face_detectors_wheights/haarcascade_frontalface_default.xml'),
                 hog=dlib.get_frontal_face_detector(),
                 dnn=cv2.dnn.readNetFromCaffe('face_detectors_wheights/deploy.prototxt', 'face_detectors_wheights/res10_300x300_ssd_iter_140000.caffemodel')
                 ):
        self.haar = haar
        self.hog = hog
        self.dnn = dnn

    def is_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        if x1 > x2 + w2 or x2 > x1 + w1:
            return False
        if y1 > y2 + h2 or y2 > y1 + h1:
            return False
        return True

    def detect_faces(self, img_BGR, show=False):
        gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)  # Convert into grayscale
        gray = cv2.equalizeHist(gray)   # equalize Histogram

        if self.haar:
            haar_boxes = self.haar.detectMultiScale(gray, 1.25, 6, minSize=(200, 200))
            haar_boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in haar_boxes])
            if show:
                for (startX, startY, endX, endY) in haar_boxes:
                    cv2.rectangle(img_BGR, (startX, startY), (endX, endY), (0, 0, 255), 10)

        if self.hog:
            faces_hog = self.hog(gray)
            hog_boxes = []
            for face in faces_hog:
                hog_boxes.append([face.left(),face.top(), face.right(), face.bottom()])
                if show:
                    cv2.rectangle(img_BGR,(face.left(),face.top()),(face.right(), face.bottom()), (255, 0, 0), 20)

        if self.dnn:
            (h, w) = img_BGR.shape[:2]
            blob = cv2.dnn.blobFromImage(img_BGR, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
            self.dnn.setInput(blob)            # Input the blob into the network
            dnn_detections = self.dnn.forward()            # Perform the forward pass to get the output of the output layer
            average_confidence = np.average(dnn_detections[0, 0, :, 2])
            dnn_boxes = []
            for i in range(0, min(30,dnn_detections.shape[2])): # assuming there are no images with more then 30 people
                confidence = dnn_detections[0, 0, i, 2]
                if confidence > 0.2: # or confidence > 3*average_confidence:    # Filter out weak detections by ensuring the 'confidence' is greater than a minimum threshold
                    # Compute the (x, y)-coordinates of the bounding box for the object
                    dnn_boxes.append((dnn_detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int"))
                    (startX, startY, endX, endY) = dnn_boxes[-1]
                    if show:
                        cv2.rectangle(img_BGR, (startX, startY), (endX, endY), (0, 255, 0), 30)
            dnn_boxes = np.array(dnn_boxes)

        if show:
            cv2.imshow('img', img_BGR)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # delete haar detections that ovelaps with dnn
        if self.haar:
            boxes_to_remove = []
            for i,haar_box in enumerate(haar_boxes):
                for dnn_box in dnn_boxes:
                    if self.is_overlap(haar_box, dnn_box):
                        boxes_to_remove.append(i)
            haar_boxes = np.delete(haar_boxes, boxes_to_remove, axis=0)

        # delete hog detections that ovelaps with dnn
        if self.hog:
            boxes_to_remove = []
            for i,hog_box in enumerate(hog_boxes):
                for dnn_box in dnn_boxes:
                    if self.is_overlap(hog_box, dnn_box):
                        boxes_to_remove.append(i)
            hog_boxes = np.delete(hog_boxes, boxes_to_remove, axis=0)

        non_empty_boxes = [boxes for boxes in [haar_boxes, hog_boxes, dnn_boxes] if boxes.size>0]
        all_boxes = []
        if non_empty_boxes:
            all_boxes = np.concatenate(non_empty_boxes , axis=0)
        return all_boxes



