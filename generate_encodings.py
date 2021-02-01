import os
import pickle
import face_recognition as fr 
# Directory containing folders of images of person to identify
KNOWN_FACES_DIR = "known_faces"
# Empty list to store the label and face encoding of images
known_faces_encoding = []
known_faces_name = []
###### Generate face encodings from the known_faces dataset ######
print("loading known faces and extracting faces features.....")
# loop through KNOWN_FACES_DIR
for name in os.listdir(KNOWN_FACES_DIR):
    # loop through folders within KNOWN_FACE_DIR
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(filename)
        image = fr.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        # Check if face encoding is valid, then append the list of names and encoding of the first face
        encoding = fr.face_encodings(image)
        if len(encoding)>0:
            known_faces_encoding.append(encoding[0])
            known_faces_name.append(name)
            print("Face found and encoding generated! \n")
        else:
            print('No face found in image!\n')

print('[INFO] Finished Training ......')
print('[INFO] Serialize Encodings into pickle file ........')
# Save the training data as PKL file
# A PKL file is a file created by pickle, a Python module that enables objects to be serialized to files on disk and deserialized back into the program at runtime.
with open('known_faces_feature.pkl','wb') as f:
    pickle.dump(known_faces_encoding,f)
    pickle.dump(known_faces_name,f)
print('[INFO] Complete .......')
