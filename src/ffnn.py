import random

import numpy as np
import math


rng = np.random.default_rng()


class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    def dim(self):
        return self.data.shape


class loss:
    def get_loss(self, input, target):
        raise NotImplementedError

    def get_gradient(self, input, target):
        raise NotImplementedError


class MSE(loss):
    def get_loss(self, input, target):
        # TODO
        return super().get_loss(input, target)

    def get_gradient(self, input, target):
        # TODO
        return super().get_gradient(input, target)


class CrossEntropyLoss(loss):
    def get_loss(self, input, target):
        # TODO
        return super().get_loss(input, target)

    def get_gradient(self, input, target):
        # TODO
        return super().get_gradient(input, target)


class init:
    @staticmethod
    def zeros(tensor: Tensor):
        tensor.data[:] = np.zeros(tensor.dim())

    @staticmethod
    def uniform(tensor: Tensor, lower_bound, upper_bound, seed):
        # TODO
        pass

    @staticmethod
    def normal(tensor: Tensor, mean, variance, seed):
        # TODO
        pass


class Layer:
    def forward(self, x: Tensor):
        raise NotImplementedError

    def backward(self, grad, lr):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor(np.empty(out_features, in_features))
        self.bias = Tensor(np.empty(out_features)) if bias else None
        init.zeros(self.weights)

    def forward(self, x):
        # TODO
        return super().forward(x)

    def backward(self, grad, lr):
        # TODO
        return super().backward(grad, lr)


class Relu(Layer):
    def forward(self, x):
        # TODO
        return super().forward(x)

    def backward(self, grad, lr):
        # TODO
        return super().backward(grad, lr)


class Sigmoid(Layer):
    def forward(self, x: Tensor) -> Tensor:
        
        self.output = 1 / (1 + np.exp(-x))

        return Tensor(self.output)

    def backward(self, grad: np.ndarray, lr) -> np.ndarray:
        
        sigmoid_deriv = self.output * (1 - self.output)

        return_grad = grad * sigmoid_deriv

        return return_grad


class Tanh(Layer):

    def forward(self, x: Tensor) -> Tensor:

        self.output = np.tanh(x.data)
        return Tensor(self.output)

    def backward(self, grad: np.ndarray, lr) -> np.ndarray:
        
        tanh_deriv = 1.0 - (self.output**2)
        return_grad = grad * tanh_deriv
        
        return return_grad


class Softmax(Layer):
    def forward(self, x: Tensor) -> Tensor:
        
        # safely shift the logits, we don't want something like e^100 to happen or something
        # e^big_num probably wouldn't happen, we normalize the data in eda after all
        # shifting the logits by the max value to prevent overflow
        # e^-big_num is much more preferred, as it gets rounded to 0
        # reminder: in softmax we only care about the total of every element to be 1

        shifted_logit = x.data - np.max(x.data, axis=1, keepdims=True)

        exps = np.exp(shifted_logit)

        self.sum_of_exp = np.sum(exps, axis=1, keepdims=True)

        self.output = exps / self.sum_of_exp

        return Tensor(self.output)

    def backward(self, grad: np.ndarray, lr) -> np.ndarray:
        
        # I have no idea whats happening here

        sum_of_grads_dot_output = np.sum(grad * self.output, axis=1, keepdims=True)

        return_grad = self.output * (grad - sum_of_grads_dot_output)

        return return_grad


class Model:
    def __init__(self, layers: list[Layer], loss: loss):
        self.layers = layers
        self.loss = loss

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def fit(self, X, y, epochs=10, batch_size=32, lr=0.01, verbose=1):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = Tensor(X[batch_idx])
                y_batch = Tensor(y[batch_idx])
                # forward prop
                y_pred = self.forward(X_batch)

                # backprop
                grad = self.loss.get_gradient(y_pred.data, y_batch.data)
                self.backward(grad, lr)

            if verbose:
                print(
                    f"Epoch {epoch+1}, Loss: {self.loss.get_loss(y_pred.data,y_batch.data):.6f}"
                )

    def predict(self, X):
        X_tensor = Tensor(X)
        out = self.forward(X_tensor)
        return np.argmax(out.data, axis=1)

    def show_weights(self, layer_idx: list[int]):
        # TODO
        pass

    def show_gradients(self, layer_idx: list[int]):
        # TODO
        pass
