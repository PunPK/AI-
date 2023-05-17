import cv2
import face_recognition
import time

face_cascade = cv2.CascadeClassifier("eye_de_mo\haarcascade_frontalface_default.xml") 

eye_cascade = cv2.CascadeClassifier("eye_de_mo\haarcascade_eye_tree_eyeglasses.xml")  
  
# capture frames from a camera 
cap = cv2.VideoCapture(2) 

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open the webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ตั้งค่าเฟรมเรทให้กล้องเว็บแคม
cap.set(cv2.CAP_PROP_FPS, 60)

# ตรวจสอบความละเอียดที่กำหนดและเฟรมเรทปัจจุบัน
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print("Resolution:", width, "x", height)
print("FPS:", fps)

current_fps = cap.get(cv2.CAP_PROP_FPS)
print("Current FPS:", current_fps)

# Set the desired FPS value
desired_fps = 60

# Set the desired FPS of the webcam
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Print the updated FPS of the webcam
updated_fps = cap.get(cv2.CAP_PROP_FPS)
print("Updated FPS:", updated_fps)

def facerectangle(faces2) :
    for (top, right, bottom, left) in faces2:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

start_time = time.time()
frame_count = 0
  
# loop runs if capturing has been initialized. 
while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
    frame_count += 1

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces2 = face_recognition.face_locations(img)
    facerectangle(faces2)
    #for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    if time.time() - start_time >= 1:
        fps = frame_count / (time.time() - start_time)
        print("Processing FPS :", round(fps, 2))
        start_time = time.time()
        frame_count = 0

    # Wait for Esc key to stop 
    k = cv2.waitKey(5)
    if k == 27: 
        break

    cv2.imshow('Ai_sleepdiver',img) 
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()