import pickle
import argparse
import cv2
import os
import warnings
import sys

# Following code block is to stop showing tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
import tensorflow as tf
sys.path.insert(0, '..')
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
    import logging
    tf.get_logger().setLevel(logging.ERROR)

import mtcnn
from FaceRecognizer import FaceRecognizer
from FaceRegister import FaceRegister


def recognize(img="img_path"):
    with open("id_list.pkl", 'rb') as f1:
        id_list = pickle.load(f1)

    img = cv2.imread(img)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    detector = mtcnn.MTCNN()
    recognizer = FaceRecognizer()

    faces = detector.detect_faces(img)
    x, y, width, height = faces[0]['box']
    img = img[y:y + height, x:x + width]
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    target_representation = recognizer.represent(img)
    min_distance_index = recognizer.verify_candidate(target_representation)
    print(f"The person is recognized as: {id_list[min_distance_index]}")


def register():
    face_register = FaceRegister()
    face_register.register_face()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['registration', 'recognition'])
    args = parser.parse_args()

    if args.mode == 'registration':
        register()
    elif args.mode == 'recognition':
        img_path = input("Please enter Image path you want to test:  ")
        recognize(img_path)
