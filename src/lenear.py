import numpy as np

from .module import Parameter


class Linear:
    def __init__(self, input_channels: int, output_channels: int, bias=True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bias = bias
        self.backward_list = []
        if bias:
            Parameter(
                [
                    self,
                    np.random.uniform(
                        -0.5, 0.5, size=(self.input_channels, self.output_channels)
                    ),
                    np.random.uniform(-0.5, 0.5, size=self.output_channels),
                ]
            )
        else:
            Parameter(
                [
                    self,
                    np.random.uniform(
                        -0.5, 0.5, size=(self.input_channels, self.output_channels)
                    ),
                    np.zeros(self.output_channels),
                ]
            )

    def __call__(self, x):
        self.x = np.array(x, copy=True)
        result = x @ Parameter.calling[self][0] + Parameter.calling[self][1]
        return result

    def backward(self, input_matrix):
        x_gradient = input_matrix @ self.weight.T
        self.weight_gradient = self.x.T @ input_matrix
        self.bias_gradient = input_matrix.mean(axis=0)
        return x_gradient
