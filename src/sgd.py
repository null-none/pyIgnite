class SGD:
    def __init__(self, model, learning_rate):
        self.model = model
        self.lr = learning_rate

    def step(self):

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ == "Linear":
                weight, bias = self.model._constructor_Parameter.calling[layer]
                weight_gradient, bias_gradient = (
                    layer.backward_list[0],
                    layer.backward_list[1],
                )
                self.model._constructor_Parameter.calling[layer] = [
                    weight - lr * weight_gradient,
                    bias - lr * bias_gradient,
                ]
