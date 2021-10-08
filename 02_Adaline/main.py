import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from Adaline_SGD import AdalineSGD

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                           random_state=8)
y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.


# TODO: Normalitzar les dades
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# TODO: Entrenar
adaline = AdalineSGD(eta=0.001, n_iter=100, shuffle=True, random_state=None)
x = adaline.fit(X, y)
y_prediction = adaline.predict(X)

# TODO: Mostrar els resultats
plt.figure(1)
z = X[:, 0]
plt.scatter(X[:, 0], X[:, 1], c=y)

minim = np.min(X[:, 0])
maxim = np.max(X[:, 0])

first_coordinate = (minim, (-adaline.w_[0]-(adaline.w_[1]*minim)) / adaline.w_[2])
second_coordinate = (maxim, (-adaline.w_[0]-(adaline.w_[1]*maxim)) / adaline.w_[2])

plt.plot([first_coordinate[0], second_coordinate[0]], [first_coordinate[1], second_coordinate[1]])

plt.figure(2)
plt.plot(adaline.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Erorr quadratic mitja')
plt.show()

print(adaline.w_)