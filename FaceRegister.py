import pickle

import cv2
from Vgg16 import Vgg16
from utils import normalize_input
import os
from retinaface import RetinaFace
from FaceRecognizer import FaceRecognizer
import mtcnn


class FaceRegister:
    def __init__(self):
        self.registration_path = "Registration_Images"

    def register_face(self):
        detector = mtcnn.MTCNN()
        representation_list = []
        id_list = {}
        for i, person in enumerate(os.listdir(self.registration_path)):
            print(person, i)
            for img in os.listdir(os.path.join(self.registration_path, person)):
                if img.endswith(".jpg") or img.endswith(".JPG"):
                    img_path = os.path.join(self.registration_path, person, img)
                    print(img_path)
                    try:
                        img = cv2.imread(img_path)
                        faces = detector.detect_faces(img)
                        if len(faces) > 0:
                            x, y, width, height = faces[0]['box']
                            img = img[y:y + height, x:x + width]
                            representation = FaceRecognizer.represent(img)
                            representation_list.append(representation)
                            id_list[i] = str(person)
                        else:
                            raise ValueError(f"No faces found for ID: {person}! "
                                  f"\nPlease try with different registration image")

                    except Exception as e:
                        print(e)
                        raise ValueError("Couldn't perform registration")
                else:
                    raise ValueError(f"The image format for ID: {person} is not supported!"
                          f"\nPlease try with 'jpg' image ")
        try:
            with open("representations.pkl", 'wb') as pkl_file:
                pickle.dump(representation_list, pkl_file)
        except Exception as e:
            print(e)

        try:
            with open("id_list.pkl", 'wb') as pkl_file:
                pickle.dump(id_list, pkl_file)
                print("Registration Successful!")
        except Exception as e:
            print(e)
