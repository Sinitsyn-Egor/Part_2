# Аномалии
import numpy as np  # noqa: E402, F401
import pandas as pd  # noqa: E402, F401

data = pd.read_csv(
    "data_files/creditcard.csv"
)  # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
print(data.head())

legit = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

X = data.drop(columns=["Time", "Class"])
y = data["Class"]

from sklearn.model_selection import train_test_split  # noqa: E402, F401

X_tr, X_tst, y_tr, y_tst = train_test_split(
    X,
    y,
    test_size=0.25,
)

from sklearn.linear_model import LogisticRegression  # noqa: E402, F401

model1 = LogisticRegression()
model1.fit(X_tr, y_tr)

import matplotlib.pyplot as plt  # noqa: E402, F401
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402, F401

# ConfusionMatrixDisplay.from_estimator(
#     model1,
#     X_tst,
#     y_tst,
#     display_labels=['Легитимная', 'Мошенническая']
# )
# plt.show() # 14

from sklearn.metrics import precision_score, recall_score  # noqa: E402, F401

y_pred = model1.predict(X_tst)
# Точность
print(precision_score(y_tst, y_pred))

# Полнота
print(recall_score(y_tst, y_pred))

# Специфичность
print(recall_score(y_tst, y_pred, pos_label=0))


# from sklearn.ensemble import RandomForestClassifier # noqa: E402, F401

# model2 = RandomForestClassifier(n_estimators=10)
# model2.fit(X_tr, y_tr)
# ConfusionMatrixDisplay.from_estimator(
#     model2,
#     X_tst,
#     y_tst,
#     display_labels=['Легитимная', 'Мошенническая']
# )
# plt.show() #

# exit()
# from sklearn.ensemble import GradientBoostingClassifier # noqa: E402, F401
# model3 = GradientBoostingClassifier()
# model3.fit(X_tr, y_tr)
# ConfusionMatrixDisplay.from_estimator(
#     model3,
#     X_tst,
#     y_tst,
#     display_labels=['Легитимная', 'Мошенническая']
# )
# plt.show() #
