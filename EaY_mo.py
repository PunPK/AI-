import cv2
import tensorflow
import keras
from PIL import Image
import face_recognition

#=----------------webcam----------------=
webcam = cv2.VideoCapture(2)
#print(webcam)
#print(webcam.read())

face_cascade = "haarcascade_frontalface_default.xml"

while 1:
    success, image_bgr = webcam.read()
 
    image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(face_cascade)
    faces = face_classifier.detectMultiScale(image_bw)

    print(f'There are {len(faces)} faces found.')

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image_bgr)
    cv2.waitKey(1)


