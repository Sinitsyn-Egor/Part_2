# субграфики
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import matplotlib as mpl  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

x = np.linspace(0, 1, 20)

# ax1 = plt.axes()
# ax1.plot(np.sin(x))
# # нижний, левый, ширина, высота
# # 0.4 - 40 % ширины рисунка
# ax2 = plt.axes([0.4, 0.3, 0.2, 0.1])
# ax2.plot(np.cos(x))

# fig = plt.figure()
# ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# ax2 = fig.add_axes([0.4, 0.3, 0.2, 0.1])
# ax1.plot(np.sin(x))
# ax2.plot(np.cos(x))

# ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
# ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
# ax1.plot(np.sin(x))
# ax2.plot(np.cos(x))

# for i in range(1, 7):
#     ax = fig.add_subplot(2, 3, i)
#     ax.plot(np.sin(x + np.pi/4*i))

# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
#
# x1 = np.linspace(0, 10, 50)
# x2 = np.linspace(0, 20, 100)
#
# for i in range(2):
#     for j in range(3):
#         if i%2==0:
#             ax[i, j].plot(np.sin(x1 + np.pi/4*(i*2 + j)))
#         else:
#             ax[i, j].plot(np.sin(x2 + np.pi/4*(i*2 + j)))

grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.1)
#   0 1 2
# 0 X Y Y
# 1 Z Z K

# plt.subplot(grid[0, 0])
# plt.subplot(grid[0, 1:])
# plt.subplot(grid[1, :2])
# plt.subplot(grid[1, 2])

#   0 1 2
# 0 X Y K
# 1 Z Z K

# plt.subplot(grid[0, 0])
# plt.subplot(grid[0, 1:])
# plt.subplot(grid[1, :2])
# plt.subplot(grid[1, 2])


# grid = plt.GridSpec(4, 4, wspace=0.2, hspace=0.2)

# Z X X X
# Z X X X
# Z X X X
#   Y Y Y

# rng = np.random.default_rng(1)
# x, y = rng.multivariate_normal([0, 0], [[1, 2], [3, 4]], 1000).T

# main_axes = plt.subplot(grid[:-1, 1:]) # X
# y_axes = plt.subplot(grid[:-1, 0]) # Y
# x_axes = plt.subplot(grid[-1, 1:]) # Z

# main_axes.plot(x, y, 'ok', alpha = 0.2)
# y_axes.hist(y, 40, orientation='horizontal', color='grey')
# y_axes.invert_xaxis()

# x_axes.hist(x, 40, color = 'grey')

# https://github.com/jakevdp/data-CDCbirths/blob/2971def79a36d299e116a96fcf774e1e505ff3ea/births.csv
# births = pd.read_csv("data_files/births.csv")
# births.dropna(inplace=True)
# print(births['day'].info())
# births["day"] = births["day"].astype(int)
#
# births.index = pd.to_datetime(
#     births["year"] * 10_000 + births["month"] * 100 + births["day"], format="%Y%m%d"
# )
#
# print(births)
#
# births_dom = births.pivot_table('births', index=[births.index['month'], births.index['day']])
#
# #births_dom.index[]
#
# fig, ax = plt.subplots()
# births_dom.plot(ax=ax)


fig = plt.figure(0)
ax1 = plt.axes()
ax1.set_xlim(0, 2)
ax2 = plt.axes([0.4, 0.3, 0.1, 0.2])

ax1.text(0.6, 0.8, "#1_1", transform=ax1.transData)
ax2.text(0.6, 0.8, "#2_1", transform=ax2.transData)

ax1.text(0.5, 0.1, "#1_2", transform=ax1.transData)
ax2.text(0.5, 0.1, "#2_2", transform=ax2.transData)

ax1.text(0.05, 0.05, "#1_3", transform=fig.transFigure)
ax2.text(0.2, 0.2, "#2_3", transform=fig.transFigure)
plt.show()
