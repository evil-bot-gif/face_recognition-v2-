import face_recognition as fr
import cv2 
import pickle
import argparse
import os
import math
import csv
import numpy as np
from datetime import datetime
import time 


"""------------Main program--------------"""
# Construct argparse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="video src url or onboard camera")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

## Global variables
Encodings = []
Names = []
# Font for text display on screen during run time
font = cv2.FONT_HERSHEY_DUPLEX
# Video Input Source
SRC= args["input"]
# Face detection model (CNN or hog) used by face recognition api
MODEL = args["detection_method"] 
# Tune accuracy of face recognition api
TOLERANCE = 0.4
# used to record the time when we processed last frame 
prev_frame_time = 0
# used to record the time at which we processed current frame 
new_frame_time = 0

"""---------------Functions-----------------"""
# Function that calculates confidence level(acc) based on euclidean distance
def face_distance_to_conf(face_distance, face_match_threshold=TOLERANCE):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

# Load trained data
with open('known_faces_feature.pkl','rb') as f:
    Encodings = pickle.load(f)
    Names= pickle.load(f)

print('\n[INFO] starting video stream ......')
# create a cam instance 
cam = cv2.VideoCapture(SRC) # ip webcam: https://192.168.0.238:4747/video (home) https://192.168.0.238:8080/video (5G router) https://192.168.0.40:18888/video (5G cellular network)
# check if the camera is open
if not cam.isOpened():
    print("Cannot open camera")
    exit()

# Instruction to quit program
print('[INFO] Press q to quit the program ........' )

while True:
    # Local variables
    known_face_names = []
    face_names = []
    # Read every frame 
    ret , frame = cam.read()
    # if the frame is not grabbed, then we reach the end of stream
    if not ret:
        break

    # Display instruction to close program at the screen
    cv2.putText(frame,'Press q to quit.',(7,80), font, 0.75,(0,255,0),2,cv2.FILLED)

    # Calculate fps
    new_frame_time = time.time()   
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    # Display FPS at the Screen
    cv2.putText(frame, f'{round(fps,1)}', (7, 40), font, 0.75, (0, 255, 0), 2, cv2.FILLED)

    # Resize frame to 1/4 of original size to speed up processing
    frameSmall = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    # Convert the BGR frame captured by Opencv to RGB for processing with face recognition api
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    # Detect the faces found in the frames of the video stream
    facePositions = fr.face_locations(frameRGB, model = MODEL)
    # Convert all the face found in the video stream into Encodings 
    allEncoding = fr.face_encodings(frameRGB,facePositions)

    # For each encoding of the detected faces, compare it with the list of known encodings (Face Recognition Process)
    for face_encoding in allEncoding:
        matches = fr.compare_faces(Encodings,face_encoding,tolerance = TOLERANCE)
        # If detected face is unknown, name label is unknown
        name = "Unknown"
        # Default acc of face recognition for unknown
        acc =100
        # Calculates the euclidean distance of the face detected with the known encodings
        face_distances = fr.face_distance(Encodings,face_encoding)
        # Retreive the best match index of face in face_distances list
        best_match_index = np.argmin(face_distances)
        # Euclidean distance of best match index 
        Euclidean_dist_best_match = face_distances[best_match_index]

        if matches[best_match_index]:
            # Name of the best match face 
            name = Names[best_match_index]
            # List of known faces detected
            known_face_names.append(name)
            # Calculate accuracy of face detection
            conf = face_distance_to_conf(Euclidean_dist_best_match)
            acc = conf * 100
        # Append the names of the detected face
        face_names.append(name)
        
    # Display the number of known faces detected
    cv2.putText(frame, f'Known_detected:{len(known_face_names)}', (400, 40), font, 0.75, (0, 255, 0), 2, cv2.FILLED)
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
        # # display the acc the accuracy 
        cv2.putText(frame, f'{round(acc,1)}%', (right, y), font, 0.5, (0, 255, 0), 2)
    # Output the frame 
    cv2.imshow('Live Video',frame)
    cv2.moveWindow('Live Video',0,0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()