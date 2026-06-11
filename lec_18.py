# Классификация
# 1. Загрузка изображения
# 2. Масштабирование
# 3. Нормализация
# 4. Выбор модели
# 5. Загрузка изображения в модель и получение предсказания
# https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
from tensorflow.keras.preprocessing import image # type: ignore
import matplotlib.pyplot as plt # noqa: E402, F401

img_path = 'data_files/cat.png'
img = image.load_img(img_path, target_size = (224, 224))


import numpy as np  # noqa: E402, F401
# plt.imshow(img)
# plt.show()

img_array = image.to_array(img)
print(img.shape)

print(img_array[100, 100])


print(np.min(img_array))
print(np.max(img_array))

img_batch = np.expand_dims(img_array)
from tensorflow.keras.application.resnet50 import preprocess_input # type: ignore # noqa: E402, F401


img_preprocessed = preprocess_input(img_batch)
print(img_preprocessed.shape)

print(img_preprocessed[0, 100, 100])

print(np.min(img_preprocessed))
print(np.max(img_preprocessed))

from tensorflow.keras.application.resnet50 import ResNet50  # type: ignore # noqa: E402

model = ResNet50()

prediction = model.predict(img_preprocessed)

print(prediction)


