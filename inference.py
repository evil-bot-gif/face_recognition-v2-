import face_recognition as fr
import cv2 
import pickle
import os
import math
import csv
import numpy as np
from datetime import datetime
import time 

from face_recognition.api import face_distance, face_encodings 

"""---------------Functions-----------------"""
# Function that provides confidence level based on euclidean distance
def face_distance_to_conf(face_distance, face_match_threshold=0.5):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def Attendance(name):
    with open('Attendance_Register.csv','r+') as f:
        DataList = f.readlines()
        names = []
        for data in DataList:
            ent = data.split(',')
            names.append(ent[0])
        if name not in names:
            curr = datetime.now()
            dt = curr.strftime('%d/%b/%Y, %H:%M:%S')
            f.writelines(f'\n{name},{dt}')


"""------------Code-------------"""
Encodings = []
Names = []
font = cv2.FONT_HERSHEY_DUPLEX
MODEL = 'CNN' 
TOLERANCE = 0.5
# GSTREAMER_IP = 'rtspsrc location=rtsp://192.168.0.238:8080/h264_pcm.sdp ! rtph264depay ! h264parse ! avdec_h264 ! decodebin ! videoconvert! appsink '
# GSTREAMER_CSI = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=720, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# GSTREAMER_TEST = "rtspsrc location=rtsp://192.168.43.1:18888/h264_pcm.sdp lateny = 30 ! decodebin ! format=(String)NV12! nvvidconv ! appsink"
# FFMPEG = 'ffmpeg -rtsp_flags listen -i rtsp://192.168.43.1:18888/h264_pcm.sdp output'

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

with open('known_faces_feature.pkl','rb') as f:
    Encodings = pickle.load(f)
    Names= pickle.load(f)

# create a cam instance 
cam = cv2.VideoCapture('http://192.168.0.238:8080/video') # ip webcam: https://192.168.0.238:4747/video (home) https://192.168.0.238:8080/video (5G router) https://192.168.0.40:18888/video (5G cellular network)

# check if the camera is open
if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Read every frame 
    known_face_names = []
    face_names = []
    ret , frame = cam.read()
    # if the frame is not grabbed, then we reach the end of stream
    if not ret:
        break

    # Calculate fps
    new_frame_time = time.time()   
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    # Display FPS at the Screen
    cv2.putText(frame, f'{round(fps,1)}', (7, 40), font, 0.75, (0, 255, 0), 2, cv2.FILLED)

    
    frameSmall = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions = fr.face_locations(frameRGB, model = MODEL)
    allEncoding = fr.face_encodings(frameRGB,facePositions)
    for face_encoding in allEncoding:
        matches = fr.compare_faces(Encodings,face_encoding,tolerance = TOLERANCE)
        name = "Unknown"
        acc =100
        # check the known faces with the smallest distance to the new face
        face_distances = fr.face_distance(Encodings,face_encoding)
        # Take the best one
        best_match_index = np.argmin(face_distances)
        # Name of the best match face
        # name = Names[best_match_index]
        # Attendance(name)
        # Euclidean distance of best match index 
        Euclidean_dist_best_match = face_distances[best_match_index]
        if matches[best_match_index]:
            name = Names[best_match_index]
            known_face_names.append(name)
            # Calculate accuracy of face detection
            conf = face_distance_to_conf(Euclidean_dist_best_match)
            acc = conf * 100
        face_names.append(name)
    # Display the number of known faces detected
    cv2.putText(frame, f'Known_detected:{len(known_face_names)}', (500, 40), font, 0.75, (0, 255, 0), 2, cv2.FILLED)
    # Draw the bounding boxes around the identified faces
    for (top,right,bottom,left), name in zip(facePositions,face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a boxes around the face and detected_name_list face label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top -15 > 15 else top + 15
        cv2.putText(frame,name.title(),(left,y),font,0.5,(0,255,0),2)
        # display the acc the accuracy 
        cv2.putText(frame, f'{round(acc,1)}%', (right, y), font, 0.5, (0, 255, 0), 2)
    # Output the frame 
    cv2.imshow('Live Video',frame)
    cv2.moveWindow('Live Video',0,0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
# print(attendance)
cam.release()
cv2.destroyAllWindows()