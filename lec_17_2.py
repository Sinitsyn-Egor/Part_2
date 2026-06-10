import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

iris = sns.load_dataset("iris")

# print(iris.head())

species_int = []
for row in iris.values:
    match row[4]:
        case "setosa":
            species_int.append(1)
        case "versicolor":
            species_int.append(2)
        case "virginica":
            species_int.append(3)

# species_int_df = pd.DataFrame(species_int)
# print(species_int_df.head())

data = iris[["sepal_length", "petal_length"]]
data["species"] = species_int

print(data.head())
print(data.shape)

data_df = data[(data["species"] == 3) | (data["species"] == 2)]
print(data.shape)

data_of_virginca = data[data["species"] == 3]
data_of_vesicolor = data[data["species"] == 2]

data_of_vesicolor_A = data_of_vesicolor.iloc[:25, :]
data_of_vesicolor_B = data_of_vesicolor.iloc[25:, :]

data_of_virginca_A = data_of_virginca.iloc[:25, :]
data_of_virginca_B = data_of_virginca.iloc[25:, :]


data_df_A = pd.concat([data_of_virginca_A, data_of_vesicolor_A], ignore_index=True)
data_df_B = pd.concat([data_of_virginca_B, data_of_vesicolor_B], ignore_index=True)


x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
)


from sklearn.tree import DecisionTreeClassifier  # noqa: E402

max_depth = [1, 3, 5, 7]

fig, ax = plt.subplots(2, 4, sharex="col", sharey="row")

X = data_df_A[["sepal_length", "petal_length"]]
y = data_df_A["species"]

j = 0
for md in max_depth:
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)

    y_p = model.predict(X_p)

    ax[0, j].scatter(
        data_of_virginca_A["sepal_length"], data_of_virginca_A["petal_length"]
    )
    ax[0, j].scatter(
        data_of_vesicolor_A["sepal_length"], data_of_vesicolor_A["petal_length"]
    )

    ax[0, j].contourf(
        X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.3, levels=[0, 2.5, 3.5]
    )
    j += 1

j = 0
for md in max_depth:
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)

    y_p = model.predict(X_p)

    ax[1, j].scatter(
        data_of_virginca_B["sepal_length"], data_of_virginca_B["petal_length"]
    )
    ax[1, j].scatter(
        data_of_vesicolor_B["sepal_length"], data_of_vesicolor_B["petal_length"]
    )

    ax[1, j].contourf(
        X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.3, levels=[0, 2.5, 3.5]
    )
    j += 1

plt.show()
