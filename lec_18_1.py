# Название папок = название категорий

Train_data_dir = 'data_files/train_data'
Validation_data_dir = 'data_files/val_data'
Train_samples = 500
Validation_samples = 500

# "кошка или собака" -> "кошка или не кошка" - бинарная классификация
# "кошка или собака" - мультиклассовая классификация
Num_classes = 2

Img_width = 224
Img_Height = 224

# Сколько изображений модель принимает одновременно
Batch_size = 64

# Аугументация - процедура увеличения кол-ва данных путём их искажения: повороты сдвиги, масштабирование

from tensorflow.keras.preprocessing import image # type: ignore # noqa: E402, F401
from tensorflow.keras.models import Model # type: ignore # noqa: E402, F401
from tensorflow.keras.preprocessing import ( # type: ignore # noqa: E402, F401
    Input,
    Faltten, 
    Dense,
    Dropout,
    GlobalAveraagePooling2D,
) 
from tensorflow.keras.application.mobienet import MobileNet, preprocess_input # type: ignore # noqa: E402, F401
from tensorflow.keras.optimizers import Adam # type: ignore # noqa: E402, F401
import math # type: ignore # noqa: E402, F401

# аугументация и нормализация
# train_datagen = image.ImageDataGenetator(
#     processing_function_input,
#     rotation_range = 2,
#     width_range = 0.2,
#     height_sft_range = 0.2,
#     zoom_range = 0.2,
# )
# только нормалиизациия
val_datagen = image.ImageDataGenerator(preprocessing_function = preprocess_input)



# target_model.save('data_files/our_model.h5')



