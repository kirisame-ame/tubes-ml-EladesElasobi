import random

import numpy as np
import math
import matplotlib.pyplot as plt
import joblib


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
        N = input.shape[0]
        y_pred = input[range(N), target]
        return np.mean(-np.log(y_pred + 1e-9))

    def get_gradient(self, input, target):
        N = input.shape[0]
        self.grad = input.copy()
        self.grad[range(N), target] -= 1
        self.grad /= N
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
        fan_in = tensor.data.shape[1] if tensor.data.ndim > 1 else tensor.data.shape[0]
        bound = math.sqrt(6.0 / fan_in)
        tensor.data[:] = np.random.uniform(-bound, bound, tensor.dim())

    @staticmethod
    def xavier_uniform(tensor: Tensor):
        fan_in = tensor.data.shape[1] if tensor.data.ndim > 1 else tensor.data.shape[0]
        fan_out = tensor.data.shape[0] if tensor.data.ndim > 1 else tensor.data.shape[0]
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        tensor.data[:] = np.random.uniform(-bound, bound, tensor.dim())


class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layer, t):
        raise NotImplementedError


class SGD(Optimizer):
    def update(self, layer, t):
        layer.weights.data -= self.lr * layer.dw
        if layer.bias is not None and layer.db is not None:
            layer.bias.data -= self.lr * layer.db


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}

    def update(self, layer, t):
        layer_id = id(layer)
        if layer_id not in self.m_w:
            self.m_w[layer_id] = np.zeros_like(layer.weights.data)
            self.v_w[layer_id] = np.zeros_like(layer.weights.data)
            if layer.bias is not None:
                self.m_b[layer_id] = np.zeros_like(layer.bias.data)
                self.v_b[layer_id] = np.zeros_like(layer.bias.data)

        self.m_w[layer_id] = (
            self.beta1 * self.m_w[layer_id] + (1 - self.beta1) * layer.dw
        )
        self.v_w[layer_id] = self.beta2 * self.v_w[layer_id] + (1 - self.beta2) * (
            layer.dw**2
        )

        m_w_hat = self.m_w[layer_id] / (1 - self.beta1**t)
        v_w_hat = self.v_w[layer_id] / (1 - self.beta2**t)

        layer.weights.data -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        if layer.bias is not None and layer.db is not None:
            self.m_b[layer_id] = (
                self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * layer.db
            )
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (
                layer.db**2
            )

            m_b_hat = self.m_b[layer_id] / (1 - self.beta1**t)
            v_b_hat = self.v_b[layer_id] / (1 - self.beta2**t)

            layer.bias.data -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class Layer:
    def forward(self, x: Tensor):
        raise NotImplementedError

    def backward(self, grad, lr):
        raise NotImplementedError

    def print_weights(self):
        if hasattr(self, "weights"):
            print(self.weights.data)
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
        init.xavier_uniform(self.weights)
        if self.bias is not None:
            init.zeros(self.bias)

    def forward(self, x):
        self.X = x
        y = x.data @ self.weights.data.T
        if self.bias is not None:
            y = y + self.bias.data
        return Tensor(y)

    def backward(self, grad, lr, reg_type=None, reg_lambda=0.0, optimizer=None, t=1):
        dw = grad.T @ self.X.data
        db = np.sum(grad, axis=0) if self.bias is not None else None
        dx = grad @ self.weights.data

        # Apply regularization to weight gradients
        if reg_type == "l1":
            dw += reg_lambda * np.sign(self.weights.data)
        elif reg_type == "l2":
            dw += 2 * reg_lambda * self.weights.data

        self.grad = dw
        self.dw = dw
        self.db = db

        if optimizer is not None:
            optimizer.update(self, t)
        else:
            self.weights.data -= lr * dw
            if self.bias is not None and db is not None:
                self.bias.data -= lr * db

        return dx


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

    def backward(self, grad, lr, reg_type=None, reg_lambda=0.0, optimizer=None, t=1):
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                grad = layer.backward(grad, lr, reg_type, reg_lambda, optimizer, t)
            else:
                grad = layer.backward(grad, lr)

    def _compute_reg_loss(self, penalty, lambda_):
        reg_loss = 0.0
        if penalty is None:
            return reg_loss
        for layer in self.layers:
            if isinstance(layer, Linear):
                if penalty == "l1":
                    reg_loss += lambda_ * np.sum(np.abs(layer.weights.data))
                elif penalty == "l2":
                    reg_loss += lambda_ * np.sum(layer.weights.data**2)
        return reg_loss

    def fit(
        self,
        X,
        y,
        epochs=10,
        batch_size=32,
        lr=0.01,
        penalty=None,
        lambda_=0.01,
        optimizer=None,
        verbose=1,
        seed=7,
        validation_data=None,
    ):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        rng = np.random.default_rng(seed=seed)

        self.optimizer = optimizer

        history = {"train_loss": [], "val_loss": []}

        t = 1
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
                self.backward(
                    grad,
                    lr,
                    reg_type=penalty,
                    reg_lambda=lambda_,
                    optimizer=self.optimizer,
                    t=t,
                )
                t += 1

            # compute full training loss
            train_pred = self.forward(Tensor(X))
            train_loss = self.loss.get_loss(
                train_pred.data, y
            ) + self._compute_reg_loss(penalty, lambda_)
            history["train_loss"].append(train_loss)

            # compute validation loss
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val, y_val = np.array(X_val), np.array(y_val)
                val_pred = self.forward(Tensor(X_val))
                val_loss = self.loss.get_loss(
                    val_pred.data, y_val
                ) + self._compute_reg_loss(penalty, lambda_)
                history["val_loss"].append(val_loss)

            if verbose:
                msg = f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}"
                if validation_data is not None:
                    msg += f", Val Loss: {history['val_loss'][-1]:.6f}"
                print(msg)

        return history

    def predict(self, X):
        X_tensor = Tensor(X)
        out = self.forward(X_tensor)
        return np.argmax(out.data, axis=1)

    def show_weights(self, layer_idx: list[int]):
        # TODO
        for idx in layer_idx:
            print(f"Layer {idx} Weights:")
            try:
                self.layers[idx].print_weights()
            except IndexError:
                print(f"Warning: Index {idx} out of range")

    def show_gradients(self, layer_idx: list[int]):
        for idx in layer_idx:
            print(f"Layer {idx} Gradients:")
            try:
                self.layers[idx].print_gradients()
            except IndexError:
                print(f"Index {idx} out of range")

    def plot_weights(self, layer_idx: list[int]):
        linear_layers = [
            (i, l)
            for i, l in enumerate(self.layers)
            if isinstance(l, Linear) and i in layer_idx
        ]
        if not linear_layers:
            print("No Linear layers found at the given indices.")
            return
        fig, axes = plt.subplots(
            1, len(linear_layers), figsize=(5 * len(linear_layers), 4)
        )
        if len(linear_layers) == 1:
            axes = [axes]
        for ax, (idx, layer) in zip(axes, linear_layers):
            w = layer.weights.data.flatten()
            ax.hist(w, bins=30, edgecolor="black", alpha=0.7)
            ax.set_title(f"Layer {idx} Weights")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        fig.suptitle("Weight Distribution")
        plt.tight_layout()
        plt.show()

    def plot_gradients(self, layer_idx: list[int]):
        linear_layers = [
            (i, l)
            for i, l in enumerate(self.layers)
            if isinstance(l, Linear) and i in layer_idx and hasattr(l, "grad")
        ]
        if not linear_layers:
            print("No Linear layers with gradients found at the given indices.")
            return
        fig, axes = plt.subplots(
            1, len(linear_layers), figsize=(5 * len(linear_layers), 4)
        )
        if len(linear_layers) == 1:
            axes = [axes]
        for ax, (idx, layer) in zip(axes, linear_layers):
            g = layer.grad.flatten()
            ax.hist(g, bins=30, edgecolor="black", alpha=0.7, color="orange")
            ax.set_title(f"Layer {idx} Gradients")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        fig.suptitle("Gradient Distribution")
        plt.tight_layout()
        plt.show()

    def save_model(self, output_filename: str):
        """
        IMPORTANT!
        in order for python to recognize the model, one of the following requirement must be fulfilled:
        - Import the model in the notebook
        - Load the model (in the notebook) within the same directory of the class model
        - Use sys.path to the class model's directory

        example on loading the model and using it:
        try:
            # Load the bundle
            bundle = joblib.load(filename)

            # Get the model and scaler
            model = bundle["model"]

            # some other preproc code

            # usage example (just call the class' function)
            preds = model.predict(X_transformed)

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            print("Run: python decision_tree.py -out_model my_model.joblib")
        except Exception as e:
            print(f"An error occurred: {e}")
        """
        bundle = {"model": self}
        joblib.dump(bundle, output_filename)
