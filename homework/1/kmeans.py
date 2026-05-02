### ЗАДАЧА НА ПРАКТИКУ № 1
# Обучение с учителем (классификация). Выбрать ДВА ЛЮБЫХ СОРТА iris и для них реализовать.
# 1. Метод главных компонент (PCA)
# 2. Случайные леса (RandomForestClassifier)

# Обучение без учителя (классификация).
# 3. Метод k средних (KMeans)

# imports
import numpy as np
import pandas as pd  # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# load|choose data
iris = sns.load_dataset("iris")
# print(iris.head())
species = ["setosa", "versicolor", "virginica"]
species_1 = species[0]
species_2 = species[2]
x_coord = 0
y_coord = 1
iris_species_mask = (iris["species"] == species_1) | (iris["species"] == species_2)
x = iris[iris_species_mask].iloc[:, [x_coord, y_coord]].to_numpy()
y = iris[iris_species_mask].iloc[:, 4].to_numpy()
# print(x)
# print(y)


# train model
model = KMeans(n_clusters=2)
model.fit(x, y)


# get|upgrade results
xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
_, Z = np.unique(Z, return_inverse=True)
Z += 1


# plot results
plt.scatter(
    x[:, 0][y == species_1],
    x[:, 1][y == species_1],
    color="red",
    alpha=0.5,
    label=species_1,
)
plt.scatter(
    x[:, 0][y == species_2],
    x[:, 1][y == species_2],
    color="green",
    alpha=0.5,
    label=species_2,
)
plt.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])
plt.xlabel(iris.columns[x_coord])
plt.ylabel(iris.columns[y_coord])
plt.legend()
plt.show()