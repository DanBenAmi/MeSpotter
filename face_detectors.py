# Import necessary libraries
import cv2
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dlib
import sys


class face_detectors:
    '''class for performing face detection using multiple methods'''
    def __init__(self,
                 # Initialize the class with pre-trained models for Haar, HOG, and DNN face detection
                 haar=cv2.CascadeClassifier(r'face_detectors_wheights/haarcascade_frontalface_default.xml'),
                 hog=dlib.get_frontal_face_detector(),
                 dnn=cv2.dnn.readNetFromCaffe('face_detectors_wheights/deploy.prototxt', 'face_detectors_wheights/res10_300x300_ssd_iter_140000.caffemodel')
                 ):
        self.haar = haar  # Haar Cascade detector
        self.hog = hog    # HOG detector from dlib
        self.dnn = dnn    # DNN detector using OpenCV's DNN module

    # Function to check if two bounding boxes overlap
    def is_overlap(self, box1, box2):
        # Extract coordinates for both boxes
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        # Return False if boxes don't overlap, True otherwise
        if x1 > x2 + w2 or x2 > x1 + w1:
            return False
        if y1 > y2 + h2 or y2 > y1 + h1:
            return False
        return True

    # Function to detect faces in an image using Haar, HOG, and DNN methods
    def detect_faces(self, img_BGR, show=False):
        # Convert image to grayscale for Haar and HOG detectors
        gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # Apply histogram equalization to improve detection

        # Detect faces using Haar Cascade
        if self.haar:
            haar_boxes = self.haar.detectMultiScale(gray, 1.25, 6, minSize=(200, 200))
            # Adjust the box format for consistency
            haar_boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in haar_boxes])
            # If show is True, draw the detected boxes on the image
            if show:
                for (startX, startY, endX, endY) in haar_boxes:
                    cv2.rectangle(img_BGR, (startX, startY), (endX, endY), (0, 0, 255), 10)

        # Detect faces using HOG
        if self.hog:
            faces_hog = self.hog(gray)
            hog_boxes = []
            for face in faces_hog:
                # Adjust the box format and draw if show is True
                hog_boxes.append([face.left(),face.top(), face.right(), face.bottom()])
                if show:
                    cv2.rectangle(img_BGR,(face.left(),face.top()),(face.right(), face.bottom()), (255, 0, 0), 20)

        # Detect faces using DNN
        if self.dnn:
            (h, w) = img_BGR.shape[:2]
            # Prepare the image as a blob for DNN
            blob = cv2.dnn.blobFromImage(img_BGR, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
            self.dnn.setInput(blob)
            dnn_detections = self.dnn.forward()
            dnn_boxes = []
            for i in range(0, min(30,dnn_detections.shape[2])): # Limit to 30 detections
                confidence = dnn_detections[0, 0, i, 2]
                if confidence > 0.2:  # Filter detections by confidence
                    # Calculate and draw bounding box if show is True
                    box = (dnn_detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                    dnn_boxes.append(box)
                    (startX, startY, endX, endY) = box
                    if show:
                        cv2.rectangle(img_BGR, (startX, startY), (endX, endY), (0, 255, 0), 30)
            dnn_boxes = np.array(dnn_boxes)

        # Display the image with detected faces if show is True
        if show:
            cv2.imshow('img', img_BGR)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Remove overlapping detections between Haar/DNN and HOG/DNN
        # This helps in reducing false positives by keeping the most confident detections
        if self.haar:
            boxes_to_remove = []
            for i,haar_box in enumerate(haar_boxes):
                for dnn_box in dnn_boxes:
                    if self.is_overlap(haar_box, dnn_box):
                        boxes_to_remove.append(i)
            haar_boxes = np.delete(haar_boxes, boxes_to_remove, axis=0)

        if self.hog:
            boxes_to_remove = []
            for i,hog_box in enumerate(hog_boxes):
                for dnn_box in dnn_boxes:
                    if self.is_overlap(hog_box, dnn_box):
                        boxes_to_remove.append(i)
            hog_boxes = np.delete(hog_boxes, boxes_to_remove, axis=0)

        # Combine the boxes from all detectors, after filtering out duplicates
        non_empty_boxes = [boxes for boxes in [haar_boxes, hog_boxes, dnn_boxes] if boxes.size > 0]
        all_boxes = np.concatenate(non_empty_boxes, axis=0) if non_empty_boxes else []

        return all_boxes  # Return all detected boxes
