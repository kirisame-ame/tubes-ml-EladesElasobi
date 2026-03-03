from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
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
model = ffnn.Model(
    layers=[ffnn.Linear(4, 4), ffnn.Relu(), ffnn.Linear(4, 3), ffnn.Softmax()],
    loss=ffnn.CrossEntropyLoss(),
)
model.fit(X_train_transformed, y_train, epochs=1000, lr=1)
preds = model.predict(X_test_transformed)

print(accuracy_score(y_test, preds))
print("actual:")
print(np.ravel(y_test))
print("preds:")
print(preds)
