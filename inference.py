import face_recognition as fr
import cv2 
import pickle
import math
import csv
import numpy as np
import time 

from face_recognition.api import face_distance, face_encodings 

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
            curr = time.datetime.now()
            dt = curr.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')


"""------------Code-------------"""
Encodings = []
Names = []
detected_name_list =[]
font = cv2.FONT_HERSHEY_DUPLEX
MODEL = 'hog' 

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

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
    # Read every frame 
    ret , frame = cam.read()
    frameSmall = cv2.resize(frame,(0,0),fx=0.33,fy=0.33)
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    # Calculate fps
    new_frame_time = time.time()   
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    # Display FPS at the Screen
    cv2.putText(frame, f'{round(fps,1)}', (7, 40), font, 1, (0, 255, 0), 2, cv2.FILLED)
    # Display the number of known faces detected
    cv2.putText(frame, f'Known_detected:{len(detected_name_list)}', (340, 40), font, 1, (0, 255, 0), 2, cv2.FILLED)
    facePositions = fr.face_locations(frameRGB, model = MODEL)
    allEncoding = fr.face_encodings(frameRGB,facePositions)
    face_names = []
    # json_to_export = {} 
    acc = 100.0
    for face_encoding in allEncoding:
        matches = fr.compare_faces(Encodings,face_encoding)
        name = 'Unknown'
        # check the known faces with the smallest distance to the new face
        face_distances = fr.face_distance(Encodings,face_encoding)
        # Take the best one
        best_match_index = np.argmin(face_distances)
        # Name of the best match face
        name = Names[best_match_index]
        # Euclidean distance of best match index 
        Euclidean_dist_best_match = face_distances[best_match_index]
        # Calculate accuracy of the detection
        conf = face_distance_to_conf(Euclidean_dist_best_match)
        acc = conf * 100
        print(acc)
        # record the attendance of the best_match face
        if matches[best_match_index]:
            # store an instance of the name of detected known face into a list
            if name not in detected_name_list:
                detected_name_list.append(name)
            print(detected_name_list)
            face_names.append(name)
            # append the time, name and date of the detected_name_list face to a json dict
            # json_to_export['Name'] = name
            # json_to_export['Time'] = f'{time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}'
            # json_to_export['Date'] = f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}'
            # Write the information of the detected_name_list face into a csv
            with open(f'Attendance/{name.title()}_{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}_attendance.csv',mode='w') as csv_file:
                    fieldnames = ['Name','Time','Date']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({'Name': name.title() , 'Time': f'{time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}', 'Date' : f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}'})

    # Draw the bounding boxes around the identified faces
    for (top,right,bottom,left), name in zip(facePositions,face_names):
        top *= 3
        right *= 3
        bottom *= 3
        left *= 3 
        
        # Draw a boxes around the face and detected_name_list face label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top -15 > 15 else top + 15
        cv2.putText(frame,name.title(),(left,y),font,0.5,(0,255,0),2)
        # display the acc the accuracy 
        cv2.putText(frame, f'{round(acc,1)}%', (right, y), font, 0.5, (0, 255, 0), 2)
    # Output the frame 
    cv2.imshow('Live Video',frame)
    cv2.moveWindow('Live Video',0,0)
    if cv2.waitKey(1) == ord('q'):
        break
# print(attendance)
cam.release()
cv2.destroyAllWindows()