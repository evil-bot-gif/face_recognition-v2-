import os
import pickle
import face_recognition as fr 
# Directory containing folders of images of person to identify
KNOWN_FACES_DIR = "known_faces"
# Empty list to store the label and face encoding of images
known_faces_encoding = []
known_faces_name = []
###### Training face_recognition to identify faces you want ######
print("loading known faces and extracting faces features.....")
# loop through KNOWN_FACES_DIR
for name in os.listdir(KNOWN_FACES_DIR):
    # loop through folders within KNOWN_FACE_DIR
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(filename)
        image = fr.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        # Encode the first face found in image
        encoding = fr.face_encodings(image)[0]
        print(encoding)
        known_faces_encoding.append(encoding)
        known_faces_name.append(name)
print(known_faces_name)

# Save the training data as PKL file
# A PKL file is a file created by pickle, a Python module that enables objects to be serialized to files on disk and deserialized back into the program at runtime.
with open('known_faces_feature.pkl','wb') as f:
    pickle.dump(known_faces_encoding,f)
    pickle.dump(known_faces_name,f)

