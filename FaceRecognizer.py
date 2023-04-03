import pickle
import utils
import cv2
import numpy as np
from torchvision.transforms import functional
from Vgg16 import Vgg16


class FaceRecognizer:
    def __init__(self):
        self.representation_list = "representations.pkl"
        self.similarity_metric = "cosine"
        self.similarity_threshold = 0.8

    @staticmethod
    def verify_candidate(target_representation):
        with open("representations.pkl", 'rb') as f1:
            representation_list = pickle.load(f1)
        representation_list = np.array(representation_list)

        # calculate cosine similarity
        similarity = utils.find_cosine_similarity_modified(representation_list, target_representation)

        similarity = np.squeeze(similarity)
        max_similarity_index = np.argmax(similarity, axis=-1)
        return max_similarity_index

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
