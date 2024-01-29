import cv2
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

def DB_to_FaceDB(DB_path, FaceDB_path, face_img_size=None):

    # Load the face detector
    face_cascade = cv2.CascadeClassifier(r'../haarcascade_frontalface_default.xml')

    for root, dirs, files in os.walk(DB_path):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Read the input image
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Convert into grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                # Detect faces
                faces_coords = face_cascade.detectMultiScale(gray, 1.25, 6)
                for (x, y, w, h) in faces_coords:
                    face = Image.fromarray(img).crop((x, y, x + w, y + h))
                    face.save(FaceDB_path)



if __name__=='__main__':
    DB_to_FaceDB('../DataBase/ImageDB/reserve', '../DataBase/FaceDB')


