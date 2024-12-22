# Aygaz-Goruntu-Isleme

# Kaggle verisini Google Colab'e bağlama
!pip install kagglehub
import kagglehub

# Veri setini indirme
rrebirrth_animals_with_attributes_2_path = kagglehub.dataset_download('rrebirrth/animals-with-attributes-2')
print('Veri seti indirildi ve bağlandı.')

# Gerekli diğer kütüphaneler
try:
    from tqdm.notebook import tqdm
except ImportError:
    !pip install tqdm
    from tqdm.notebook import tqdm

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Veri setinin dizini
base_path = "/root/.cache/kagglehub/datasets/rrebirrth/animals-with-attributes-2/versions/1/Animals_with_Attributes2/JPEGImages/"

# Kullanılacak hayvan sınıfları
animals = ["collie", "dolphin", "elephant", "fox", "moose", "rabbit", "sheep", "squirrel", "giant+panda", "polar+bear"]

# Resim yollarını organize etme
image_paths = {}
for dirname, _, filenames in os.walk(base_path):
    for animal in animals:
        if animal in dirname:
            if animal not in image_paths:
                image_paths[animal] = []
            for filename in filenames:
                image_paths[animal].append(os.path.join(dirname, filename))

# Her sınıf için maksimum 650 resim al
max_images_per_class = 650
image_data = []
labels = []

for label, animal in enumerate(animals):
    paths = image_paths[animal][:max_images_per_class]
    for path in paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (128, 128))  # Tüm resimleri 128x128 boyutuna yeniden boyutlandır
        image_data.append(image / 255.0)  # Normalize et
        labels.append(label)

image_data = np.array(image_data)
labels = np.array(labels)

# Eğitim ve test verilerine bölme
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.3, random_state=42)
y_train = to_categorical(y_train, num_classes=len(animals))
y_test = to_categorical(y_test, num_classes=len(animals))

print("Veri seti hazırlandı.")



data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
data_generator.fit(X_train)






model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(animals), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])











history = model.fit(
    data_generator.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=15
)







test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

if test_accuracy < 0.5:
    print("Model doğruluğu düşük! Parametreleri veya model yapısını değiştirip yeniden deneyiniz.")




def manipulate_images(images):
    manipulated = []
    for img in images:
        bright_img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)  # Parlaklık artırma
        manipulated.append(bright_img)
    return np.array(manipulated)

manipulated_test = manipulate_images(X_test)

# Manipüle edilmiş test setini değerlendirme
manipulated_test_loss, manipulated_test_accuracy = model.evaluate(manipulated_test, y_test)

print(f"Manipüle edilmiş Test Seti: Test Loss: {manipulated_test_loss}, Test Accuracy: {manipulated_test_accuracy}")



def apply_gray_world(images):
    result = []
    for img in images:
        mean_r = np.mean(img[:, :, 0])
        mean_g = np.mean(img[:, :, 1])
        mean_b = np.mean(img[:, :, 2])
        avg_mean = (mean_r + mean_g + mean_b) / 3
        img[:, :, 0] = img[:, :, 0] * avg_mean / mean_r
        img[:, :, 1] = img[:, :, 1] * avg_mean / mean_g
        img[:, :, 2] = img[:, :, 2] * avg_mean / mean_b
        result.append(img)
    return np.array(result)

color_corrected_test = apply_gray_world(manipulated_test)
color_corrected_test_loss, color_corrected_test_accuracy = model.evaluate(color_corrected_test, y_test)

print(f"Renk Sabitliği Uygulanmış Test Seti: Test Loss: {color_corrected_test_loss}, Test Accuracy: {color_corrected_test_accuracy}")
