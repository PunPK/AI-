import face_recognition
import cv2
from mouth_open_algorithm import get_lip_height, get_mouth_height
from datetime import datetime

def is_mouth_open(face_landmarks):
    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']

    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)
    
    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.5
    #print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f' 
    #      % (top_lip_height,bottom_lip_height,mouth_height, min(top_lip_height, bottom_lip_height) * ratio))
    #cv2.rectangle(frame,(mouth_height),(0,127,255),2) 
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return True
    else:
        return False
    
print("-------------------------------------------")
print("------------start Ai_sleepdiver------------")
print("-------------------------------------------")

#face_cascade = cv2.CascadeClassifier("eye_de_mo\haarcascade_frontalface_default.xml") 

eye_cascade = cv2.CascadeClassifier("eye_de_mo\haarcascade_eye_tree_eyeglasses.xml")  
  
# capture frames from a camera 
cap = cv2.VideoCapture(1) 
  
# loop runs if capturing has been initialized. 
while 1:  
  
    # reads frames from a camera 
    ret, frame = cap.read()  
    face_locations = face_recognition.face_locations(frame)
    #face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(frame)
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces2 = face_recognition.face_locations(frame)
    #for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
    #for face_landmarks in zip(face_landmarks_list):
        #ret_mouth_open = is_mouth_open(face_landmarks)
        #if ret_mouth_open is True:
        #    text = 'Mouth is open'
        #else:
        #    text = 'Open your mouth'
        #print(text)
        #i = 1
    #for (top, right, bottom, left) in faces2:
    for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    #for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        ret_mouth_open = is_mouth_open(face_landmarks)
        if ret_mouth_open is True:
            text = 'Mouth is open' 
        else:
            text = 'Mouth is close'
        print(text)
        roi_gray = gray[top:top+bottom, left:left+right] 
        roi_color = frame[top:top+bottom, left:left+right] 
        #roi_gray = gray[y:y+h, x:x+w] 
        #roi_color = img[y:y+h, x:x+w] 
  
        # Detects eyes of different sizes in the input image 
        eyes = eye_cascade.detectMultiScale(roi_gray)  
  
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
  
    # Display an image in a window 
    cv2.imshow('Ai_sleepdiver',frame) 
  
    k = cv2.waitKey(5) # Wait Esc to stop 
    if k == 27: 
        print("-------------------------------------------")
        print("------------close Ai_sleepdiver------------")
        print("-------------------------------------------")
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()