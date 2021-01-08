from imutils.video import VideoStream
from datetime import datetime
import imutils
import face_recognition as fr
import argparse
import cv2 
import time
import os


# Construct argparse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-i", "--input",required=True,
	help="video src url for IP webcam")

args = vars(ap.parse_args())

# if output directory not available make directory
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])


# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print('[INFO] starting video stream ......')
vs = VideoStream(src= args["input"]).start()
time.sleep(2.0)
total = 0

# Instruction for using the program
print('[INFO] Press k while program is running to take a Snapshot ........' )
print('[INFO] Press q to quit the program and tabulate the amount of image taken ........' )
# loop over the frames from the video stream
while True: 
    # grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk)
    frame = vs.read()
    orig = frame.copy()
    

    # Convert frame from BGR to RGB, for face recognition api to process
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rects = fr.face_locations(frameRGB, model = args["detection_method"])

    # draw the bounding box for face detected 
    for (top,right,bottom,left) in rects:
	    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # show the output frame 
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'k' key was pressed, write the "original" frame to disk
    # so we can later process it and used it for face recognition 
    if key == ord('k') :
        # get the basename of the folder where your images are stored in 
        base_name = os.path.basename(args["output"])
        # Get current time and date 
        curr = datetime.now()
        dt = curr.strftime('%d-%b-%Y_%H%M%S')
        # Join the directory with the image file name
        p = os.path.sep.join([args["output"],f"{base_name}_{dt}.png"])
        cv2.imwrite(p,orig)
        total += 1
        print(f'{total} Image taken.')

    # if the 'q' key was pressed, break from the loop 
    elif key == ord("q"):
        break

# print the total faces saved and do a bit of cleanup
print(f"[INFO] {total} face images stored ")
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
   