import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# IMAGE SETTINGS
IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 32

# ✅ Replace this path with your local dataset folder
dataset_path = r"C:\Users\janvi\OneDrive\Desktop\Prodigy\TASK - 5\food_dataset"

# FOOD → CALORIE MAPPING
calorie_map = {
    "apple": 52,
    "banana": 89,
    "pizza": 266,
    "burger": 295,
    "salad": 152
}

# DATA LOADER
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(dataset_path, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                    batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
val = datagen.flow_from_directory(dataset_path, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                  batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# CNN MODEL
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=5, validation_data=val)

model.save("food_classifier_model.h5")
print("✅ Model Saved as food_classifier_model.h5")

# INFERENCE
class_labels = list(train.class_indices.keys())

def estimate_calories(pred):
    food = class_labels[np.argmax(pred)]
    cal = calorie_map.get(food, "Unknown")
    return food, cal

# SAMPLE PREDICTION
img_batch, _ = next(val)
pred = model.predict(img_batch[:1])
food, cal = estimate_calories(pred)

plt.imshow(img_batch[0])
plt.axis('off')
plt.title(f"{food} - {cal} kcal")
plt.show()
