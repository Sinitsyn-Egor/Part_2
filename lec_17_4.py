import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: E402, F401
import seaborn as sns

iris = sns.load_dataset("iris")

data = iris[["sepal_length", "petal_length", "species"]]
data_setosa = data[data["species"] == "setosa"]

print(data_setosa.head())

X = data_setosa["sepal_length"]
Y = data_setosa["petal_length"]

data_setosa = data_setosa.drop(columns=["species"])

print(data_setosa.head())


from sklearn.decomposition import PCA  # noqa: E402, F401

pca = PCA(n_components=2)
pca.fit(data_setosa)

print("1")

print(pca.components_)

print(pca.mean_)

print(pca.explained_variance_)


plt.scatter(X, Y)

plt.scatter(*pca.mean_)

# 1
plt.plot(
    [
        pca.mean_[0],
        pca.mean_[0] + pca.components_[0, 0] * np.sqrt(pca.explained_variance_[0]),
    ],
    [
        pca.mean_[1],
        pca.mean_[1] + pca.components_[0, 1] * np.sqrt(pca.explained_variance_[0]),
    ],
)

# 2
plt.plot(
    [
        pca.mean_[0],
        pca.mean_[0] + pca.components_[1, 0] * np.sqrt(pca.explained_variance_[1]),
    ],
    [
        pca.mean_[1],
        pca.mean_[1] + pca.components_[1, 1] * np.sqrt(pca.explained_variance_[1]),
    ],
)


pcal = PCA(n_components=1)
pcal.fit(data_setosa)

X_pcal = pcal.transform(data_setosa)

print(data_setosa.shape)
print(X_pcal.shape)

X_new = pcal.inverse_transform(X_pcal)

plt.scatter(X_new[:, 0], X_new[:, 1])

plt.show()

# - аномальные значения
