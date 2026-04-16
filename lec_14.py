import numpy as np
import pandas as pd  # noqa: F401
import matplotlib.pyplot as plt


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))


# x = np.linspace(-6, 6, 30)
# y = np.linspace(-10, 10, 50)
#
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
#
#
fig = plt.figure()
ax = plt.axes(projection="3d")
#
# # ax.scatter3D(X, Y, X, c = Z)
# # ax.plot_wireframe(X, Y, Z)
# ax.plot_surface(X, Y, Z, cmap = 'viridis')

# angle = np.linspace(0, 2*np.pi, 50)
# r = np.linspace(0, 6, 30)
#
# R, Angle = np.meshgrid(r, angle)
#
# X = R * np.sin(Angle)
# Y = R * np.cos(Angle)
# Z = f(X, Y)
#
# ax.plot_surface(X, Y, Z, cmap = 'viridis')


# angle = np.linspace(0, 1.5 * np.pi, 50)
# r = np.linspace(0, 6, 30)
#
# R, Angle = np.meshgrid(r, angle)
#
# X = R * np.sin(Angle)
# Y = R * np.cos(Angle)
# Z = f(X, Y)
#
# ax.scatter3D(X, Y, X, c = Z)
#
# # ax.plot_surface(X, Y, Z, cmap = 'viridis')
# ax.plot_trisurf(X, Y, Z, cmap = 'viridis')
#
#
# plt.show()


# Seaborn

import seaborn as sns  # noqa: E402, F401

sns.set_style("darkgrid")
cars = pd.read_csv("data_files/cars.csv")

print(cars.head())

# Числовые данные

# парная

# sns.pairplot(cars)
#
# sns.pairplot(data=cars, hue = 'transmission')
#
# # Тепловая карта
#
#
# cars_corr = cars[['year', 'selling_price', 'seats', 'mileage']]
#
# sns.heatmap(cars_corr.corr(), cmap = 'viridis', hue='fuel')
#
# # Д рассеяния
# sns.scatterplot(x='seats', y='mileage', data=cars, hue='fuel')
# sns.scatterplot(x='year', y='selling_price', data=cars)
#
# # Д.рассеяния + лин.регрессия
# sns.regplot(x='seats', y='mileage', data=cars)
#
# sns.regplot(x='seats', y='mileage', data=cars, kind='scatter')


# sns.regplot(x='seats', y='mileage', data=cars, kind='scatter')

# Линейнай график
# sns.lineplot(s='seats', y='mileage', data=cars, hue='fuel')


# Сводная диаграмма

# sns.jointplot(x='year', y='selling_price', data=cars)
#
# sns.jointplot(x='year', y='selling_price', data=cars, kind='kde')
#
# sns.jointplot(x='year', y='selling_price', data=cars, kind='hex')
#
# sns.jointplot(x='year', y='selling_price', data=cars, kind='transmission')


# Критерии и числа

# barplot catplot boxplot violinplot
plt.show()
