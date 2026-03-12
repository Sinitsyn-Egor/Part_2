import numpy as np
import pandas as pd

print(pd.date_range("2026-01-01", periods=4, freq="ME"))

print(pd.date_range("2026-01-01", periods=4, freq="MS"))

print(pd.date_range("2026-01-01", periods=4, freq="QE"))

print(pd.date_range("2026-01-01", periods=4, freq="QS"))

print(pd.date_range("2026-01-01", periods=4, freq="W"))

print(pd.date_range("2026-01-01", periods=4, freq="4W-MON"))

# yhoo/finance
ind = pd.read_csv("data_files/index.csv", sep=";", parse_dates=["Date"])

print(ind.head())

print(type(ind))
print(ind.dtypes)

index = pd.DatetimeIndex(ind["Date"])

ind.index = index
ind = ind["Close"]
print(ind.head())

# import matplotlib.pyplot as plt  # noqa: E402
# ind.plot()
# plt.show()

# https://github.com/jakevdp/bicycle-data/blob/main/FremontBridge.csv
df = pd.read_csv(
    "data_files/FremontBridge.csv",
    index_col="Date",
    parse_dates=True,
    date_format="%m/%d/%Y %I:%M:%S %p",
)
print(df.head())
print(df.dtypes)

print(df.columns)
df.columns = ["Total", "East", "West"]
print(df.head())

print(df.describe())
print(df.dropna().describe())


import matplotlib.pyplot as plt  # noqa: E402, F401

# df.plot()
# plt.title('Кол-во велосипедистов (в час)')
# plt.show()

# weekly = df.resample('W').sum()
# weekly.plot(style = ['-', ':', '--'])
# plt.ylabel('Кол-во велосипедистов (в час)')
# plt.show()

# daily = df.resample('D').sum()
# daily.rolling(30, center=True, win_type='gaussian').mean(std=5).plot(style = ['-', ':', '--'])
# plt.ylabel('Среднее месячное кол-во велосипедистов (в час)')
# plt.show()


# timely = df.groupby(df.index.time).mean()
# ticks = 60*60*4*np.arange(6)
# # print(ticks)
# timely.plot()
# plt.show()

# weekly = df.groupby(df.index.dayofweek).mean()
# weekly.plot()
# plt.show()

# w1 = np.where(df.index.weekday < 5, 'Будни', 'Выходные')
# t1 = df.groupby([w1, df.index.time]).mean()
#
# fig, ax = plt.subplots(1, 2)
# t1.loc['Будни'].plot(ax = ax[0], title='Будни')
# t1.loc['Выходные'].plot(ax = ax[1], title='Выходные')
# plt.show()


# MATPLOTLIB

# pip install matplotlib
# pip install pyQt5

import matplotlib.pyplot as plt  # noqa: E402, F811

plt.style.use("classic")
plt.plot(np.arange(12))
plt.show()
