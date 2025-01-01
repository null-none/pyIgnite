import numpy as np

from .module import Parameter


class CrossEntropyLoss:

    def __init__(self):
        self.predicted = None
        self.true = None

    def __call__(self, logits, true):
        predicted = np.exp(logits) / np.sum(np.exp(logits), axis=1).reshape(-1, 1)
        self.predicted = np.array(predicted, copy=True)
        self.true = np.array(true, copy=True)
        self.number_of_classes = predicted.shape[1]
        self.true = np.array(true, copy=True)
        self.loss = -1 * np.sum(true * np.log(predicted + 1e-5), axis=1)
        return self

    def backward(self):
        loss = self.predicted - self.true
        for index, layer in enumerate(Parameter.layers[::-1]):
            if type(layer).__name__ == "Linear":
                changes_w = (layer.x.T @ loss) / loss.shape[0]
                if layer.bias:
                    changes_b = np.sum(loss) / loss.shape[0]
                else:
                    changes_b = 0
                layer.backward_list = [changes_w, changes_b]
                loss = loss @ Parameter.calling[layer][0].T
            elif type(layer).__name__ == "ReLU":
                loss = layer.backward(loss)
