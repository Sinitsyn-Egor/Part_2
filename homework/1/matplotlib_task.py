import numpy as np
import matplotlib.pyplot as plt


def plot_1():
    x = [2, 5, 10, 15, 20]
    y1 = [1, 7, 3, 5, 11]
    y2 = [4, 3, 1, 8, 12]
    plt.plot(x, y1, "-or", label="line 1")
    plt.plot(x, y2, "-.og", label="line 1")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_2():
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    x = [1, 2, 3, 4, 5]
    y1 = [1, 7, 6, 4, 5]
    y2 = [9, 4, 2, 4, 9]
    y3 = [-7, -4, 2, -4, -7]
    ax1.plot(x, y1)
    ax2.plot(x, y2)
    ax3.plot(x, y3)
    plt.tight_layout()
    plt.show()


def plot_3():
    x = np.linspace(-5, 5, 11)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, x**2)
    ax.arrow(
        0,
        9.5,
        0,
        -8.5,
        head_length=1,
        fc="green",
        ec="black",
        width=0.1,
        linewidth=1,
    )
    ax.text(0, 10, "min")
    plt.tight_layout()
    plt.show()


def plot_4():  # defoult_rng(1) ?
    rng = np.random.default_rng(1)
    colors = 10 * rng.random((7, 7))
    plt.imshow(colors, vmin=0, vmax=10, extent=(0, 7, 0, 7))
    cax = plt.axes([0.85, 0.11, 0.04, 0.385])  # [left, bottom, width, height]
    col = plt.colorbar(cax=cax)  # noqa: F841
    # print(col.ax)
    plt.show()


def plot_5():  # alpha = 0.7 ?
    x = np.linspace(0, 5, 100)
    y = np.cos(np.pi * x)
    plt.plot(x, y, "r")
    plt.fill_between(x, y, 0, color="blue", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_6():  # 1 fill не справляется почему-то
    x = np.linspace(0, 5, 100)
    y = np.cos(np.pi * x)
    plt.plot(x, y, linewidth=2.5)
    plt.fill_between(x, -1, -0.5, where=(y <= -0.5), color="white", zorder=10)
    plt.fill_between(np.linspace(0, 4.7, 100), -0.8, -0.5, color="white", zorder=10)
    plt.xlim(-0.2, 4.8)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()


def plot_7():
    x = np.arange(7)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), subplot_kw={"aspect": "equal"})
    for ax in axes:
        ax.plot(x, x, "og")
        ax.grid()
    axes[0].plot(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        "g",
    )
    axes[1].plot(
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        "g",
    )
    axes[2].plot(
        [0, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 5.5, 5.5, 6],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        "g",
    )
    plt.show()


def plot_8():
    x = np.linspace(0, 10, 11)
    y1 = x * (10 - x) / 25 * 5
    y2 = x * (10 - x) / 25 * 15
    y3 = x * (14 - x) / 49 * 25
    plt.fill_between(x, 0, y1, color="blue", label="y1")
    plt.fill_between(x, y1, y2, color="orange", label="y2")
    plt.fill_between(x, y2, y3, color="green", label="y3")
    plt.legend(loc="upper left")
    plt.show()


def plot_9():  # распределение процентов
    brands = ["Toyota", "BMW", "AUDI", "Jaguar", "Ford"]
    market_share = [10, 21, 12, 8, 8]
    colors = ["orange", "green", "red", "purple", "blue"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.pie(
        market_share,
        labels=brands,
        colors=colors,
        startangle=60,
        explode=[0, 0.1, 0, 0, 0],
    )
    plt.tight_layout()
    plt.show()


def plot_10():
    brands = ["Toyota", "BMW", "AUDI", "Jaguar", "Ford"]
    market_share = [10, 21, 12, 8, 8]
    colors = ["orange", "green", "red", "purple", "blue"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.pie(
        market_share,
        labels=brands,
        colors=colors,
        startangle=60,
        wedgeprops={"width": 0.5},
    )
    plt.tight_layout()
    plt.show()


plots = [
    plot_1,
    plot_2,
    plot_3,
    plot_4,
    plot_5,
    plot_6,
    plot_7,
    plot_8,
    plot_9,
    plot_10,
]

for plot in plots:
    plot()
