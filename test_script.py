import cv2 
import time
import math
import pickle
import face_recognition as  fr

# Static variable

GSTREAMER_IP = 'rtspsrc location=rtsp://192.168.0.238:8080/h264_pcm.sdp ! rtph264depay ! h264parse ! avdec_h264 ! decodebin ! videoconvert! appsink '
font = cv2.FONT_HERSHEY_DUPLEX 
MODEL = 'CNN'
# Dynamic variable

# used to record the time when we processed last frame 
prev_frame_time = 0
# used to record the time at which we processed current frame 
new_frame_time = 0

Encodings = []
Names = []


# _____main_____ #

# Load the encodings and name label into the list
with open('known_faces_feature.pkl','rb') as f:
    Encodings = pickle.load(f)
    Names= pickle.load(f)

# open camera 
cap = cv2.VideoCapture()
cap.open('https://192.168.0.238:8080/video')

# check if the camera is open
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    # if frame not grabbed stop program
    if not ret:
        break
    # Calculate fps
    new_frame_time = time.time()   
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    # Display FPS at the Screen
    cv2.putText(frame, f'{round(fps,1)}', (7, 40), font, 1, (0, 255, 0), 2, cv2.FILLED)

    # resize frame to 1/4 for faster processsing 
    frameSmall = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    # Find the position of faces within that frame and convert all the detected face into encodings
    facePositions = fr.face_locations(frameRGB,model = MODEL)
    allEncodings = fr.face_encodings(frameRGB,facePositions)

    # for each face detected and their corresponding encodings compare with know faces encodings
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        # Label for unknown face detected 
        name = 'Unknown'
        # compare the faces detected with known face find the label of the matched face
        matches = fr.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]

        # Rescale the frame to original size 
        top *= 4 
        right *= 4
        bottom *= 4 
        left *= 4
        # Draw a boxes around the face and detected_name_list face label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top -15 > 15 else top + 15
        cv2.putText(frame,name.title(),(left,y),font,0.5,(0,255,0),2)

    cv2.imshow("camCapture",frame)
    # Scan for all key press
    key = cv2.waitKey(1) & 0xFF
    # if 'q' pressed, break the loop   
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()