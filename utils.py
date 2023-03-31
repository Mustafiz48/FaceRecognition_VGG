import numpy as np
from numpy import dot


def normalize_input(img):
    """
        For VGG Face
    """
    img[..., 0] = img[..., 0] - 93.5940
    img[..., 1] = img[..., 1] - 104.7624
    img[..., 2] = img[..., 2] - 129.1863
    return img


"""
  distance.py
"""


def find_cosine_distance(source_representation, test_representation):
    a = dot(source_representation, test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    # return dot(source_representation, test_representation) / (norm(source_representation) * norm(test_representation))


def find_euclidean_distance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))
