# фильтрация спама
# бинарная классификация
# Векторизация

# столбцы = слова (в тексте)
# строки = образцы текста
# ячейки = кол-во данных слов в данном тексте

# очистка: строчные, удаляют знаки препинания, (стоп-слова)

import numpy as np  # noqa: E402, F401
import pandas as pd  # noqa: E402, F401

data = pd.read_csv(
    "data_files/spam.csv"
)  # https://www.kaggle.com/datasets/phangud/spamcsv
print(data.head())

data["Spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0)

print(data.columns)


from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402, F401

vectozer = CountVectorizer()
X = vectozer.fit_transform(data["Message"])
w = vectozer.get_feature_names_out()

# print(w)
# print(w[1000])

from sklearn.model_selection import train_test_split  # noqa: E402, F401

X_tr, X_tst, y_tr, y_tst = train_test_split(
    data["Message"],
    data["Spam"],
    test_size=0.25,
)

from sklearn.naive_bayes import MultinomialNB  # noqa: E402, F401

from sklearn.pipeline import Pipeline  # noqa: E402, F401

md = Pipeline([("vectorizer", CountVectorizer()), ("nb", MultinomialNB())])

md.fit(X_tr, y_tr)

texts = [
    "Hi! How are you?",  # 0
    "Win the lottery",  # 0
    "Free subscription",  # 1
    "Black Friday big discount shop offer",  # 0
    "Nice to meet you",  # 0
]

print(md.predict(texts))
