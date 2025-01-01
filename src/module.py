import numpy as np

Parameter = None


def ParameterObj():
    class Parameter:
        layers = []
        calling = dict()

        def __init__(self, info):
            Parameter.layers.append(info[0])
            Parameter.calling[info[0]] = info[1:]

    return Parameter


class Module:
    def __init__(self):
        self._constructor_Parameter = ParameterObj()
        global Parameter
        Parameter = self._constructor_Parameter

    def forward(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self
