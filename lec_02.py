import numpy as np
import timeit  # noqa: F401


# слияние и разбиение

# одномерные
x = np.array([1, 2, 3])
y = np.array([4, 5])
z = np.array([6])

xyz = np.concatenate([x, y, z])
print(xyz)

# двумерные
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
xy1 = np.concatenate([x, y])
print(xy1)

xy2 = np.concatenate([x, y], axis=0)
print(xy2)

xy3 = np.concatenate([x, y], axis=1)
print(xy3)


x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])

print(np.vstack([x, y]))

print(np.hstack([x, y]))

print(np.dstack([x, y]))


# Разбиение массивов

xy = np.vstack([x, y])
print(xy)

print(np.split(xy, [1], axis=1))

print(np.vsplit(xy, [2]))

print(np.vsplit(xy, [2]))


z = np.dstack([x, y])
print(z)

print(np.dsplit(z, [1]))

# Универсальные функции

x = np.arange(1, 10)
print(x)


def f(x):
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = 1.0 / x[i]
    return out


print(f(x))
print(1.0 / x)

# print(timeit.timeit(stmt = 'f(x)', globals=globals()))
# print(timeit.timeit(stmt = '1.0/x', globals=globals()))


# УФ арифметические операции


x = np.arange(5)
print(x)

print(x + 1)
print(x - 1)
print(x * 2)
print(x / 2)
print(x // 2)
print(-x)
print(x**2)
print(x % 2)

print(x * 2 - 2)


print(x + 1)
print(np.add(x, 1))


x = np.arange(-5, 4)
print(x)

print(abs(x))
print(np.abs(x))
print(np.absolute(x))

x = np.array([3 + 4j, 4 - 3j])
print(abs(x))
print(np.abs(x))

# УФ тригонометрические
# считай все

# УФ показатели и логарифмы
# тоже много

x = [0, 0.001, 0.0001, 0.01, 0.1]
print("exp = ", np.exp(x))

print("exp-1 = ", np.expm1(x))

print("log(x) = ", np.log(x))

print("log(x+1) = ", np.log1p(x))

# УФ куча всего и scipy

x = np.arange(5)
print(x)
y = x * 10
print(y)
y = np.multiply(x, 10)
print(y)

z = np.empty(len(x))
np.multiply(x, 10, out=z)
print(y)


x = np.arange(5)
z = np.zeros(10)
print(x)
print(z)
z[::2] = x * 10

print(z)

z = np.zeros(10)
np.multiply(x, 10, out=z[::2])
print(z)

# Сводные показатели

x = np.arange(1, 5)
print(x)
print(np.add.reduce(x))
print(np.add.accumulate(x))


print(np.multiply.reduce(x))
print(np.multiply.accumulate(x))

print(np.subtract.reduce(x))
print(np.subtract.accumulate(x))

print(np.sum(x))
print(np.cumsum(x))
print(np.prod(x))
print(np.cumprod(x))

x = np.arange(1, 10)
print(np.add.outer(x, x))

print(np.multiply.outer(x, x))

# Агрегирование данных

np.random.seed(2)
s = np.random.random(100)
print(sum(s))
print(np.sum(s))

a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(sum(a))
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))

print(type(a))
print(a.sum())
print(a.sum(0))
print(a.sum(1))

print(sum(a, 2))


# Минимум и максимум
np.random.seed(1)
s = np.random.random(100)

print(min(s))
print(np.min(s))

print(max(s))
print(np.max(s))


# mean std varm median argmin argmax percentile any all
# nan*

# Not a number - NaN

# транслирование(broadcaasting)

a = np.array([1, 2, 3])
b = np.array([5, 5, 5])

print(a + b)
# [6, 7, 8]

print(a + 5)
