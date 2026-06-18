import numpy as np  # noqa: E402, F401
import pandas as pd  # noqa: E402, F401

data = pd.read_csv(
    "data_files/phishing.csv"
)  # https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector
print(data.head())

print(data.columns)

X = data.drop(columns=["class"])
print(X.columns)

y = pd.DataFrame(data["class"])
print(y.columns)


from sklearn.model_selection import train_test_split  # noqa: E402, F401

X_tr, X_tst, y_tr, y_tst = train_test_split(
    X,
    y,
    test_size=0.25,
)

from sklearn.tree import DecisionTreeClassifier  # noqa: E402, F401

dt = DecisionTreeClassifier()

model = dt.fit(X_tr, y_tr)

predict = model.predict(X_tst)

from sklearn.metrics import accuracy_score  # noqa: E402, F401

print(accuracy_score(predict, y_tst))

# Классификациия: бинарные(двоичные), мультиклассовые, многометочные
# - точность (precision) - стоимость ложных  срабатываний высока
# - полнота (recall) - стоимость ложноотрицательных срабатываний высока
# - спецефичность (specification) = полнота истино оположительная
# - чуствительность (sensitivity) = полнота
# - F1 - мера

# Метрики - процента ошибок, процент правильных ответов (accuracy)
# Типы ошибочных ответов: ложноположительные (ложная тревога), ложноотрицательные (ложный пропуск)
# Типы правильных ответов истиноположительные, остиноотрицательные
