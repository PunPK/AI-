import cv2 as cv
import cv2 as cv2
import mediapipe as mp
import time
import utils, math
import numpy as np
# Fast Ai
from fastbook import *
from glob import glob
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn_inf_eye = load_learner('eye_data_resnet18_fastai.pkl')
learn_inf_yawn = load_learner('yawn_data_resnet18_fastai.pkl')

# Variables 
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# Face bounder indices 
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Lips indices for landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  

map_face_mesh = mp.solutions.face_mesh
# Camera object 
camera = cv.VideoCapture(0)

# Landmark detection function 
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

def cropped_frame(frames) :
    # คำนวณตำแหน่งที่ต้องตัดวิดีโอเพื่อให้อยู่ตรงกลางของ 1080p
    width = frames.shape[1]
    height = frames.shape[0]
    
    start_x = int((width - 640) / 2)
    start_y = int((height - 480) / 2)
    end_x = start_x + 640
    end_y = start_y + 480

    # ตัดขนาดวิดีโอ
    cropped_frame = frames[start_y:end_y, start_x:end_x]

    return cropped_frame

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # List of (x,y) coordinates
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # Returning the list of tuples for each landmark 
    return mesh_coord

# Euclidean distance 
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking ratio
def detecteye(img, landmarks, right_indices, left_indices):
    # Right eye
    # Horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # Vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # Left eye 
    # Horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # Vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    # Draw lines on eyes 
    # Right eye
    eye_right_x_min = min([landmarks[idx][0] for idx in RIGHT_EYE])
    eye_right_x_max = max([landmarks[idx][0] for idx in RIGHT_EYE])
    eye_right_y_min = min([landmarks[idx][1] for idx in RIGHT_EYE])
    eye_right_y_max = max([landmarks[idx][1] for idx in RIGHT_EYE])

    # Increase width of rectangle
    width_increase = 20
    eye_right_x_min -= width_increase
    eye_right_x_max += width_increase
    eye_right_y_min -= width_increase
    eye_right_y_max += width_increase
    # Draw rectangle around right eye
    cv.rectangle(img, (eye_right_x_min, eye_right_y_min), (eye_right_x_max, eye_right_y_max), utils.GREEN, 2)

    # Left eye
    eye_left_x_min = min([landmarks[idx][0] for idx in LEFT_EYE])
    eye_left_x_max = max([landmarks[idx][0] for idx in LEFT_EYE])
    eye_left_y_min = min([landmarks[idx][1] for idx in LEFT_EYE])
    eye_left_y_max = max([landmarks[idx][1] for idx in LEFT_EYE])

    # Increase width of rectangle
    eye_left_x_min -= width_increase
    eye_left_x_max += width_increase
    eye_left_y_min -= width_increase
    eye_left_y_max += width_increase

    # Draw rectangle around left eye
    cv.rectangle(img, (eye_left_x_min, eye_left_y_min), (eye_left_x_max, eye_left_y_max), utils.GREEN, 2)

    # Crop eye regions from the image based on the rectangles
    eye_right_image = img[eye_right_y_min:eye_right_y_max, eye_right_x_min:eye_right_x_max]
    eye_left_image = img[eye_left_y_min:eye_left_y_max, eye_left_x_min:eye_left_x_max]

    # Perform prediction on cropped eye regions
    re_right = learn_inf_eye.predict(eye_right_image)
    re_left = learn_inf_eye.predict(eye_left_image)

    print("Eye right:", re_right)
    print("Eye left:", re_left)
        
# Yawn detection function 
def detectYawn(img, landmarks, LIPS):
    # Lips coordinates
    lips_points = [landmarks[idx] for idx in LIPS]

    # Find the minimum and maximum x and y coordinates of the lips points
    x_values = [point[0] for point in lips_points]
    y_values = [point[1] for point in lips_points]
    lips_x_min = min(x_values)
    lips_x_max = max(x_values)
    lips_y_min = min(y_values)
    lips_y_max = max(y_values)

    # Increase width and height of the rectangle
    width_increase = 25
    height_increase = 15
    lips_x_min -= width_increase
    lips_x_max += width_increase
    lips_y_min -= height_increase
    lips_y_max += height_increase

    # Draw rectangle around lips
    cv.rectangle(img, (lips_x_min, lips_y_min), (lips_x_max, lips_y_max), utils.GREEN, 2)

    Yawn_image = img[lips_y_min:lips_y_max, lips_x_min:lips_x_max]

    # Perform prediction on cropped eye regions
    re_yawn = learn_inf_yawn.predict(Yawn_image)
    print("Yarw :", re_yawn)


#Set_FRANE(camera)

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # Starting time
    start_time = time.time()

    # Starting video loop
    while True:
        frame_counter += 1 # Frame counter
        ret, frame = camera.read() # Get frame from camera
        if not ret:
            break # No more frames, break

        # Resize frame
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        #frame = cropped_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            # Eye and yawn detection
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            #detectEyes(frame, gray)
            detecteye(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            detectYawn(frame, mesh_coords,LIPS)

        # Calculate frame per second (FPS)
        end_time = time.time() - start_time
        fps = frame_counter / end_time

        frame = utils.textWithBackground(frame, f'FPS: {round(fps,1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

cv.destroyAllWindows()
camera.release()
