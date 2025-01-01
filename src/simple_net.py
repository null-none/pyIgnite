from .lenear import Linear
from .flatten import Flatten
from .relu import ReLU
from .softmax import Softmax
from .module import Module


class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(input_channels=25, output_channels=10, bias=True)
        self.linear2 = Linear(input_channels=10, output_channels=2, bias=True)
        self.flatten = Flatten()
        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, x):
        x_1 = self.flatten(x)
        x_2 = self.linear1(x_1)
        x_3 = self.relu(x_2)
        x_4 = self.linear2(x_3)
        return x_4
