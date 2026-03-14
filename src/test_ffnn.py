from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import numpy as np
import ffnn

iris = datasets.load_iris(as_frame=True)

# Set feature matrix X and target vector y
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=5
)
pt = PowerTransformer(standardize=True)  # Standard Scaling already included
X_train_transformed = pt.fit_transform(X_train)
X_test_transformed = pt.transform(X_test)

print("\n--- Test 1: Control (Xavier init, SGD optimizer) ---")
model = ffnn.Model(
    layers=[ffnn.Linear(4, 4), ffnn.Relu(), ffnn.Linear(4, 3), ffnn.Softmax()],
    loss=ffnn.CrossEntropyLoss(),
)
model.fit(X_train_transformed, y_train, epochs=20, lr=0.01, penalty="l2", lambda_=0.001)
preds = model.predict(X_test_transformed)

print("Accuracy:", accuracy_score(y_test, preds))

print("\n--- Test 2: Linear initialized with Kaiming ---")
layer1 = ffnn.Linear(4, 4)
ffnn.init.kaiming_uniform(layer1.weights)
layer2 = ffnn.Linear(4, 3)
ffnn.init.kaiming_uniform(layer2.weights)

model_kaiming = ffnn.Model(
    layers=[layer1, ffnn.Relu(), layer2, ffnn.Softmax()],
    loss=ffnn.CrossEntropyLoss(),
)
model_kaiming.fit(
    X_train_transformed, y_train, epochs=20, lr=0.01, penalty="l2", lambda_=0.001
)
preds_kaiming = model_kaiming.predict(X_test_transformed)

print("Accuracy:", accuracy_score(y_test, preds_kaiming))

print("\n--- Test 3: Adam optimizer ---")
model_adam = ffnn.Model(
    layers=[ffnn.Linear(4, 4), ffnn.Relu(), ffnn.Linear(4, 3), ffnn.Softmax()],
    loss=ffnn.CrossEntropyLoss(),
)
opt = ffnn.Adam()
model_adam.fit(
    X_train_transformed,
    y_train,
    epochs=20,
    lr=0.01,
    optimizer=opt,
    penalty="l2",
    lambda_=0.001,
)
preds_adam = model_adam.predict(X_test_transformed)

print("Accuracy:", accuracy_score(y_test, preds_adam))


sk = MLPClassifier(
    hidden_layer_sizes=(4, 3),
    activation="relu",
    learning_rate="constant",
    learning_rate_init=0.1,
    max_iter=20,
)
sk.fit(X_train_transformed, y_train)
skpreds = sk.predict(X_test_transformed)

print("\n--- Scikit-Learn MLPClassifier ---")
print("Accuracy:", accuracy_score(y_test, skpreds))
import matplotlib

matplotlib.use("Agg")
model.plot_weights([0, 2])
model.plot_gradients([0, 2])
