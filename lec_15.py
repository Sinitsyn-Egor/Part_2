# scikit-learn

import numpy as np  # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt  # noqa: F401

iris = sns.load_dataset("iris")
print(iris.head())

print(type(iris))

print(type(iris.values))

print(iris.values.shape)

print(iris.columns)

print(iris.index)

# sns.pairplot(iris, hue = 'species')

# plt.show()

# Строки - образцы - отдельный объект (sample)
# Столбцы - признаки (feature)
# Матрица признаков [число образцов признаков] - признаки - Независимая переменная
# Целевой массив (target, label) [1 на число обрзцов] - зависимая переменная

X_iris = iris.drop("species", axis=1)
print(X_iris)

y_iris = iris["species"]
print(y_iris)

# 1. Выбирается класс модели
# 2. Выбираются гиперпараметры модели
# 3. На основе создаётся матрица и целевой вектор
# 4. Обучение модели (fit)
# 5. Обученая модель применяется к новым данным
#  5.1. Обучение без учителя predict()
#  5.2. Обучение с учителем - predict() или transform()


# С учителем. Регрессия. Линейная регрессия.

x = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()

# 1. Выбирается класс модели
from sklearn.linear_model import LinearRegression  # noqa: E402, F401

# 2. Выбираются гиперпараметры модели
model = LinearRegression(fit_intercept=True)

# 3. На основе создаётся матрица и целевой вектор

# 4. Обучение модели (fit)
reg = model.fit(x[:, np.newaxis], y)

# plt.scatter(x, y)

# 5. Обученая модель применяется к новым данным
#  5.1. Обучение без учителя predict()

xfit = np.linspace(0, x.max(), 1000)
yfit = model.predict(xfit[:, None])

# plt.plot(xfit, yfit, 'r')

# plt.plot(xfit, xfit*reg.coef_ + reg.intercept_, 'r')

# y = kx + b


from sklearn.preprocessing import PolynomialFeatures  # noqa: E402, F401
from sklearn.pipeline import make_pipeline  # noqa: E402, F401

model = make_pipeline(PolynomialFeatures(7), LinearRegression())
reg = model.fit(x[:, np.newaxis], y)

xfit = np.linspace(x.min(), y.min(), 1_000)
yfit = model.predict(xfit[:, None])

# plt.scatter(x, y)
# plt.plot(xfit, yfit, 'r')

# plt.show()

# Класиффикация

x_0 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y_0 = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()
x_1 = iris[iris["species"] == "virginica"].iloc[:, 0].to_numpy()
y_1 = iris[iris["species"] == "virginica"].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

x_00 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
x_11 = iris[iris["species"] == "virginica"].iloc[:, 0].to_numpy()

plt.scatter(x_00, np.full(50, 1), color="red", alpha=0.5)
plt.scatter(x_11, np.full(50, 5), color="red", alpha=0.5)


from sklearn.linear_model import LogisticRegression  # noqa: E402, F401

model = LogisticRegression()

x = iris[iris["species"] != "virgincia"].iloc[:, 0].to_numpy()
print(x.shape)
y = iris[iris["species"] != "virgincia"].iloc[:, 4]
print(y.shape)
print(y)

model.fit(x[:, None], y)

xfit = np.linspace(x.min(), x.min(), 1_000)
yfit = model.predict_proba(xfit[:, None])

# print(yfit)

# plt.plot(xfit, 1 + 4 * yfit[:, 1], 'green')

# plt.plot(xfit, 1 + 4 * yfit[:, 1], 'red')
# plt.show()

# Деревья решений

from sklearn.tree import DecisionTreeClassifier  # noqa: E402, F401

x = iris[iris["species"] != "virgincia"].iloc[:, 0:2].to_numpy()
y = iris[iris["species"] != "virgincia"].iloc[:, 4].to_numpy()
y1 = np.full(50, 1)
y2 = np.full(50, 2)
print(y1)
print(type(y1))

# y = np.ravel([y1, y2])

# print(x)
print(y)

tree = DecisionTreeClassifier()
tree.fit(x, y)

print(np.c_[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])

print(np.ravel([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]))


xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

print(Z)

# ax = plt.gca()

# ax.contourf(xx, yy, Z, alpha = 0.3)


# plt.show()
