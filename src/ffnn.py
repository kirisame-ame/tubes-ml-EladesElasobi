import random

import numpy as np
import math


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
    # def _to_one_hot(self, input, target):
    #     if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1):
    #         one_hot = np.zeros((target.shape[0], input.shape[1]))
    #         one_hot[np.arange(target.shape[0]), target.flatten().astype(int)] = 1
    #         return one_hot
    #     return target

    def get_loss(self, input, target):
        pass
        # target = self._to_one_hot(input, target)
        # return np.mean((input - target) ** 2)

    def get_gradient(self, input, target):
        pass
        # target = self._to_one_hot(input, target)
        # return 2 * (input - target) / input.shape[0]


class CrossEntropyLoss(loss):
    def get_loss(self, input, target):
        y_pred = input[:, target]
        return np.mean(-np.log(y_pred + 1e-9))

    def get_gradient(self, input, target):
        # TODO
        self.grad = input.copy()
        self.grad[:, target] -= 1
        self.grad /= input.shape[0]
        return self.grad


class init:
    @staticmethod
    def zeros(tensor: Tensor):
        tensor.data[:] = np.zeros(tensor.dim())

    @staticmethod
    def uniform(tensor: Tensor, lower_bound, upper_bound, seed):
        rng = np.random.default_rng(seed=seed)
        tensor.data[:] = rng.uniform(lower_bound, upper_bound, tensor.dim())

    @staticmethod
    def normal(tensor: Tensor, mean, variance, seed):
        rng = np.random.default_rng(seed=seed)
        tensor.data[:] = rng.normal(mean, variance**0.5, tensor.dim())

    @staticmethod
    def kaiming_uniform(tensor: Tensor):
        # TODO
        pass

    @staticmethod
    def xavier_uniform(tensor: Tensor):
        pass


class Layer:
    def forward(self, x: Tensor):
        raise NotImplementedError

    def backward(self, grad, lr):
        raise NotImplementedError

    def print_weights(self):
        if hasattr(self, "weights"):
            print(self.weights)
        # else print nothing

    def print_gradients(self):
        if hasattr(self, "grad"):
            print(self.grad)
        # else print nothing


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor(np.empty((out_features, in_features)))
        self.bias = Tensor(np.empty(out_features)) if bias else None
        init.zeros(self.weights)

    def forward(self, x):
        self.X = x
        y = x.data @ self.weights.data.T
        if self.bias is not None:
            y = y + self.bias.data
        return Tensor(y)

    def backward(self, grad, lr):
        dw = grad.T @ self.X.data
        db = np.sum(grad, axis=0)
        dx = grad @ self.weights.data
        self.weights.data -= lr * dw
        if self.bias is not None:
            self.bias.data -= lr * db
        return dx


class LinearActivation(Layer):
    def forward(self, x: Tensor) -> Tensor:
        self.input = x.data
        return Tensor(x.data)

    def backward(self, grad: np.ndarray, lr) -> np.ndarray:
        return grad * 1.0  


class Relu(Layer):
    def forward(self, x):
        self.mask = (x.data > 0).astype(float)
        return Tensor(x.data * self.mask)

    def backward(self, grad, lr):
        return grad * self.mask


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

    def fit(
        self, X, y, epochs=10, batch_size=32, lr=0.01, penalty=None, verbose=1, seed=7
    ):
        n_samples = X.shape[0]
        rng = np.random.default_rng(seed=seed)
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
        for layer in self.layers[layer_idx]:
            layer.print_weights()

    def show_gradients(self, layer_idx: list[int]):
        for layer in self.layers[layer_idx]:
            layer.print_gradients()
