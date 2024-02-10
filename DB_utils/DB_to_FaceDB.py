import cv2
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dlib
import sys

# Set the current working directory to the project directory (two levels up from the current file's directory)
project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(project_dir)

# Import the face detection functionality
from face_detectors import *


def DB_to_FaceDB(DB_path, FaceDB_path, face_img_size=None):
    '''function to process an image database and extract faces into a separate database'''
    # Initialize the face detection model
    face_detector = face_detectors()

    # Traverse the directory containing images
    for root, dirs, files in os.walk(DB_path):
        for file in files:
            # Check for images with jpg or png extension
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                file_path = os.path.join(root, file)  # Full path of the image
                img_BGR = cv2.imread(file_path)  # Read image in BGR color space
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)  # Convert to RGB color space for PIL compatibility

                # Detect faces within the image
                faces = face_detector.detect_faces(img_BGR)

                # For each detected face
                for idx, (startX, startY, endX, endY) in enumerate(faces):
                    # Crop the face from the original image
                    face = Image.fromarray(img_RGB).crop((startX, startY, endX, endY))
                    # Save the cropped face image in the specified FaceDB directory with a unique name
                    face.save(os.path.join(FaceDB_path,f'{file[:-4]}_{idx}.jpg'))

# Main execution: run the DB_to_FaceDB function with paths to the source image database and the target face database
if __name__ == '__main__':
    DB_to_FaceDB('DataBase/ImageDB/reserve', 'DataBase/FaceDB')
