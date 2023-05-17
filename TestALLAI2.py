import face_recognition
import cv2
import time

def Set_FRANE(cap) :
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ตั้งค่าเฟรมเรทให้กล้องเว็บแคม
    cap.set(cv2.CAP_PROP_FPS, 60)

    # ตรวจสอบความละเอียดที่กำหนดและเฟรมเรทปัจจุบัน
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Resolution:", width, "x", height)
    print("FPS:", fps)

def face_MO(frame) :
    face_locations = face_recognition.face_locations(frame)
    #face_encodings = face_recognition.face_encodings(frame, face_locations)
    #face_landmarks_list = face_recognition.face_landmarks(frame)
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces2 = face_recognition.face_locations(frame)

    for (top, right, bottom, left) in faces2:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
        roi_gray = gray[top:top+bottom, left:left+right] 
        roi_color = frame[top:top+bottom, left:left+right] 

        # Detects eyes 
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 10)
        #eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 10)
  
        #draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 

        # Detects mouth 
        #mouth = mouth_cascade.detectMultiScale(roi_color, scaleFactor = 1.8, minNeighbors = 20)  
        mouth = mouth_cascade.detectMultiScale(roi_color, scaleFactor = 2.2, minNeighbors = 30)  
  
        #draw a rectangle in mouth
        for (mx,my,mw,mh) in mouth: 
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(120,60,222),2) 

def Updated_FPS(cap):
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Current FPS:", current_fps)

    # Set the desired FPS value
    desired_fps = 60

    # Set the desired FPS of the webcam
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # Print the updated FPS of the webcam
    updated_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Updated FPS:", updated_fps)


#=========================================================================================================#

eye_cascade = cv2.CascadeClassifier("eye_de_mo\haarcascade_eye_tree_eyeglasses.xml")  

mouth_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
  
# capture frames from a camera 
cap = cv2.VideoCapture(2) 

#Set_FRANE(cap)
#Updated_FPS(cap)

# ตัวแปรสำหรับคำนวณ FPS
start_time = time.time()
frame_count = 0

print("-------------------------------------------")
print("------------start Ai_sleepdiver------------")
print("-------------------------------------------")

# loop 
while 1:  
  
    # reads frames from a camera 
    ret, frame = cap.read()  

    face_MO(frame)

    # Display
    cv2.imshow('Ai_sleepdiver',frame) 
    frame_count += 1

    if time.time() - start_time >= 1:
        fps = frame_count / (time.time() - start_time)
        print("Processing FPS :", round(fps, 2))
        start_time = time.time()
        frame_count = 0
  
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