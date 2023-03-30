import pickle

from Vgg16 import Vgg16
import torchvision.transforms.functional as TF
import utils
import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self):
        self.representation_list = "representations.pkl"
        self.distance_metric = "cosine"
        self.distance_threshold = 0.25

    def verify_candidate(self, target_representation):
        # representations = []
        with open("representations.pkl", 'rb') as f1:
            representations = pickle.load(f1)

        representations = np.array(representations)

        target_representation = np.array(target_representation)
        target_representation = np.expand_dims(target_representation, axis=0)
        target_representation = np.transpose(target_representation)

        # calculate distance
        if self.distance_metric == 'cosine':
            distance = utils.findCosineDistance(representations, target_representation)
        elif self.distance_metric == 'eucledian':
            distance = utils.findEuclideanDistance(representations, target_representation)
        elif self.distance_metric == 'euclidean_l2':
            distance = utils.findEuclideanDistance(utils.l2_normalize(representations),
                                                   utils.l2_normalize(target_representation))

        # print(distance)
        distance = np.squeeze(distance)
        # print(distance)
        min_distandance_index = np.argmin(distance, axis=-1)
        # print(min_distandance_index)
        return min_distandance_index

    @staticmethod
    def represent(img):
        # load vgg face model
        vgg_16 = Vgg16.get_model(weights_path="weights/vgg_face_dag.pth")
        img = cv2.resize(img, (224, 224))
        img = utils.normalize_input(img=img)
        img = TF.to_tensor(img)
        img.unsqueeze_(0)

        # represent
        result = vgg_16.forward(img)
        result = result.cpu().detach().numpy()
        result = result[0]
        return result
