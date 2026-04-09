import numpy as np  # noqa: F401
import pandas as pd # noqa: F401
import matplotlib as mpl # noqa: F401
import matplotlib.pyplot as plt # noqa: F401

# Figure
# Axes - axis (x, y)

# grid = plt.GridSpec(1, 2)
# 
# ax1 = plt.subplot(grid[0, 0])
# ax1.set_xscale('log')
# ax1.set_xlim(1, 1000)
# ax1.grid(True, which='major')
# 
# ax2 = plt.subplot(grid[0, 1])
# ax2.set_yscale('log')
# ax2.set(ylim=(1, 1000))
# ax1.grid(True, which='major', axis='y')
# 
# print(ax1.xaxis.get_major_locator())
# print(ax1.xaxis.get_major_formatter())
# print(ax1.xaxis.get_minor_locator())
# print(ax1.xaxis.get_minor_formatter())
# 
# print(ax2.xaxis.get_major_locator())
# print(ax2.xaxis.get_major_formatter())
# print(ax2.xaxis.get_minor_locator())
# print(ax2.xaxis.get_minor_formatter())




# plt.show()

# from sklearn.datasets import fetch_olivetti_faces
# 
# faces = fetch_olivetti_faces().images
# 
# fig, ax = plt.subplots(7, 7)
# fig.subplots_adjust(hspace = 0, wspace = 0)
# 
# for i in range(7):
#     for j in range(7):
#         ax[i, j].xaxis.set_major_locator(plt.NullLocator())
#         ax[i, j].yaxis.set_major_locator(plt.NullLocator())
#         ax[i, j].imshow(faces[7*i+j])


# def ff(value, tick_number):
#     return tick_number
# 
# 
# x = np.linspace(0, 4*np.pi, 1000)
# 
# 
# fig, ax = plt.subplots()
# 
# ax.plot(x, np.sin(x), label = 'Sinus')
# ax.plot(x, np.cos(x), label = 'Cosinus')
# 
# ax.grid(True)
# ax.legend()
# ax.set_xlim(0, 4*np.pi)
# 
# ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi/4))
# 
# ax.xaxis.set_major_formatter(plt.FuncFormatter(ff))















# fig, ax = plt.subplots(8, 1)
# 
# x = np.linspace(0, 10, 10)
# 
# for i in ax.flat:
#     i.plot(x, x*0+2)
# 
# ax[0].xaxis.set_major_locator(plt.NullLocator())
# ax[1].xaxis.set_major_locator(plt.MultipleLocator(0.8))
# ax[2].xaxis.set_major_locator(plt.FixedLocator([1, 3, 8, 9]))
# ax[3].xaxis.set_major_locator(plt.LinearLocator(numticks=4))
# ax[4].xaxis.set_major_locator(plt.IndexLocator(base=2, offset=1.3))
# ax[5].xaxis.set_major_locator(plt.AutoLocator())
# ax[6].xaxis.set_major_locator(plt.MaxNLocator(8))
# ax[7].xaxis.set_major_locator(plt.LogLocator(base=3))
# 
# 
# 
# import matplotlib.ticker as mtik  # noqa: E402, F401
# 
# ax[1].xaxis.set_major_formatter(plt.NullFormatter())
# ax[2].xaxis.set_major_formatter(plt.FixedFormatter(['a', 'b', 'c', 'd']))
# ax[3].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f $m^2$'))
# ax[4].xaxis.set_major_formatter(plt.PercentFormatter(xmax=4))

# plt.style.use('lec_13.style')
# print(plt.style.available)
# # exit()
# 
# plt.style.use('Solarize_Light2')
# 
# x = np.random.randn(1000)
# # plt.figure(facecolor='#921212')
# # plt.axes(facecolor='#adadadad')
# # 
# # plt.rc('figure', facecolor='#921212')
# # plt.rc('axes', facecolor='#adadadad')
# 
# plt.hist(x)

from mpl_toolkits import mplot3d  # noqa: F401


def f(x, y):
    return np.sin(x*np.pi/2 + np.sqrt(x**2 + y**2))


x = np.linspace(-6, 6, 30)
y = np.linspace(-10, 10, 50)

print(x.shape)
print(y.shape)

X, Y = np.meshgrid(x, y)

print(X.shape)
print(Y.shape)

print(X)
print(Y)

Z = f(X, Y)

print(Z.shape)
print(Z)




fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 40)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(0, 90)

plt.show()


















