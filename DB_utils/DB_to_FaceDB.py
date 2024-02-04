import cv2
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dlib
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(project_dir)

from face_detectors import *

def DB_to_FaceDB(DB_path, FaceDB_path, face_img_size=None):

    face_detector = face_detectors()     # initial the face detectors

    for root, dirs, files in os.walk(DB_path):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Read the input image
                img_BGR = cv2.imread(file_path)
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

                # Detect faces
                faces = face_detector.detect_faces(img_BGR)

                for idx, (startX, startY, endX, endY) in enumerate(faces):
                    face = Image.fromarray(img_RGB).crop((startX, startY, endX, endY))
                    face.save(os.path.join(FaceDB_path,f'{file[:-4]}_{idx}.jpg'))


if __name__=='__main__':
    DB_to_FaceDB('DataBase/ImageDB/reserve', 'DataBase/FaceDB')


