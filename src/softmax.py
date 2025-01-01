import numpy as np


class Softmax:
    def __init__(self):
        pass

    def __call__(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)
