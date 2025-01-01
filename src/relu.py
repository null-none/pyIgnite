import numpy as np


class ReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, input_matrix):
        return (self.x > 0) * input_matrix
