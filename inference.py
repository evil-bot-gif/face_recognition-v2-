import face_recognition as fr
import cv2 
import pickle
import os
import math
import csv
import numpy as np
from datetime import datetime
import time 


"""---------------Functions-----------------"""
# Function that provides confidence level based on euclidean distance
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
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
# Initialize some variables
Encodings = []
Names = []
face_names = []
facePositions = []
font = cv2.FONT_HERSHEY_DUPLEX
MODEL = 'hog' 
process_this_frame = True
acc = 100
# used to record the time when we processed last frame 
prev_frame_time = 0
# used to record the time at which we processed current frame 
new_frame_time = 0

# Load the encodings and names into the list
with open('known_faces_feature.pkl','rb') as f:
    Encodings = pickle.load(f)
    Names= pickle.load(f)

# create a cam instance 
cam = cv2.VideoCapture(0) # ip webcam: https://192.168.0.238:4747/video (home) https://192.168.0.238:8080/video (5G router) https://192.168.0.40:18888/video (5G cellular network)
# check if the camera is open
if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Grab a single frame from video
    ret , frame = cam.read()

    # Resize the frame to 1/4 for faster face recognition processing
    frameSmall = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

    # Convert the image from BGR color to RGB color 
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)

    # Only process everyother frame to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        facePositions = fr.face_locations(frameRGB, model = MODEL)
        allEncoding = fr.face_encodings(frameRGB,facePositions)
        
        face_names = []
        for face_encoding in allEncoding:
            matches = fr.compare_faces(Encodings,face_encoding,tolerance = 0.5)
            name = 'Unknown'
            # check the known faces with the smallest distance to the new face
            face_distances = fr.face_distance(Encodings,face_encoding)
            # Know face with smallest distance to the new face
            best_match_index = np.argmin(face_distances)
            # record the attendance of the best_match face
            if matches[best_match_index]:
                # Name of the best match face
                name = Names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    # Display the results
    for (top,right,bottom,left), name in zip(facePositions,face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw a boxes around the face and detected_name_list face label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top -15 > 15 else top + 15
        cv2.putText(frame,name.title(),(left,y),font,0.5,(0,255,0),2)
    
    # Display the resulting image  
    cv2.imshow('Live Video',frame)
    cv2.moveWindow('Live Video',0,0)
    # Hit Q to quit
    if cv2.waitKey(1) == ord('q'):
        break
# print(attendance)
cam.release()
cv2.destroyAllWindows()