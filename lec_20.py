# Аномалии (метод лавных компанент)
import numpy as np  # noqa: E402, F401
import pandas as pd  # noqa: E402, F401

data = pd.read_csv("data_files/creditcard.csv")
print(data.head())

legit = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

legit = legit.drop(["Class", "Time"], axis=1)
fraud = fraud.drop(["Class", "Time"], axis=1)

print(legit.shape)
print(fraud.shape)

from sklearn.decomposition import PCA  # noqa: E402, F401

# 29 - 26
pca = PCA(n_components=26, random_state=0)

legin_pca = pd.DataFrame(pca.fit_transform(legit), index=legit.index)
print(legin_pca)
print(legin_pca.shape)

fraud_pca = pd.DataFrame(pca.fit_transform(fraud), index=fraud.index)
print(fraud_pca)
print(fraud_pca.shape)

legin_restore = pd.DataFrame(pca.inverse_transform(legin_pca), index=legin_pca.index)
print(legin_restore)

fraud_restore = pd.DataFrame(pca.inverse_transform(fraud_pca), index=fraud_pca.index)
print(fraud_restore)


def score(legit_orig, legit_rest):
    l = np.sum((np.array(legit_orig) - np.array(legit_rest)) ** 2, axis=1)  # noqa: E741
    return pd.Series(data=l, index=legit_orig.index)


legit_score = score(legit, legin_restore)
print(legit_score.shape)
fraud_score = score(fraud, fraud_restore)
print(fraud_score.shape)

import matplotlib.pyplot as plt  # noqa: E402, F401

# fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
# ax[0].plot(legit_score)
# ax[1].plot(fraud_score)

# plt.show()


th = 150
true_pos = fraud_score[fraud_score >= th].count()
print(true_pos)

true_neg = legit_score[legit_score < th].count()
print(true_neg)

false_pos = legit_score[legit_score >= th].count()
print(false_pos)

false_neg = fraud_score[fraud_score < th].count()
print(false_neg)

mat = [[true_neg, false_pos], [false_neg, true_pos]]

import seaborn as sns  # noqa: E402, F401

sns.heatmap(
    mat,
    xticklabels=["Легитимная", "Мошеннеческая"],
    yticklabels=["Легитимная", "Мошеннеческая"],
    fmt="d",
    annot=True,
)

plt.xlabel("Предсказание")
plt.ylabel("Истина")

plt.show()
