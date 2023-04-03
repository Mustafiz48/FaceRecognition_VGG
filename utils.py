import numpy as np
from numpy import dot
from numpy.linalg import norm


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


# This is the correct one, gives accurate result for same vector
def find_cosine_similarity_modified(source_representation, test_representation):
    # Calculate magnitude of single vector
    test_representation_mag = np.linalg.norm(test_representation)

    # Calculate magnitude of each vector in list
    source_representation_mags = np.linalg.norm(source_representation, axis=1)

    # Calculate dot product of single vector and each vector in list
    dot_products = np.dot(source_representation, test_representation)

    # Calculate cosine similarity between single vector and each vector in list
    similarities = dot_products / (test_representation_mag * source_representation_mags)
    return similarities


# This one follows formula, but for some reason doesn't give accurate result when same vector is passed
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
