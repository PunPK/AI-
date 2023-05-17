import cv2
import face_recognition
import time

face_cascade = cv2.CascadeClassifier("eye_de_mo\haarcascade_frontalface_default.xml") 

eye_cascade = cv2.CascadeClassifier("eye_de_mo\haarcascade_eye_tree_eyeglasses.xml")  
  
# capture frames from a camera 
cap = cv2.VideoCapture(2) 

start_time = time.time()
frame_count = 0
  
# loop runs if capturing has been initialized. 
while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces2 = face_recognition.face_locations(img)
    for (top, right, bottom, left) in faces2:
        ##cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    #for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[top:top+bottom, left:left+right] 
        roi_color = img[top:top+bottom, left:left+right] 
        #roi_gray = gray[y:y+h, x:x+w] 
        #roi_color = img[y:y+h, x:x+w] 
  
        # Detects eyes of different sizes in the input image 
        eyes = eye_cascade.detectMultiScale(roi_gray)  
  
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
  
    # Display an image in a window 
    cv2.imshow('img',img) 
    frame_count += 1

    if time.time() - start_time >= 1:
        fps = frame_count / (time.time() - start_time)
        print("Processing FPS :", round(fps, 2))
        start_time = time.time()
        frame_count = 0

    # Wait for Esc key to stop 
    k = cv2.waitKey(5)
    if k == 27: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()