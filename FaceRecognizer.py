import pickle
import utils
import cv2
import numpy as np
from torchvision.transforms import functional
from Vgg16 import Vgg16


class FaceRecognizer:
    def __init__(self):
        self.representation_list = "representations.pkl"
        self.distance_metric = "cosine"
        self.distance_threshold = 0.25
        self.distance = None

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
            self.distance = utils.find_cosine_distance(representations, target_representation)
        elif self.distance_metric == 'euclidean':
            self.distance = utils.find_euclidean_distance(representations, target_representation)
        elif self.distance_metric == 'euclidean_l2':
            self.distance = utils.find_euclidean_distance(utils.l2_normalize(representations),
                                                          utils.l2_normalize(target_representation))

        self.distance = np.squeeze(self.distance)
        min_distance_index = np.argmin(self.distance, axis=-1)
        return min_distance_index

    @staticmethod
    def represent(img):
        # load vgg face model
        vgg_16 = Vgg16.get_model()
        img = cv2.resize(img, (224, 224))
        img = utils.normalize_input(img=img)
        img = functional.to_tensor(img)
        img.unsqueeze_(0)

        # represent
        result = vgg_16.forward(img)
        result = result.cpu().detach().numpy()
        result = result[0]
        return result
