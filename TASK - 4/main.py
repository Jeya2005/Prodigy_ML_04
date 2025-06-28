import os
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Config
DATASET_DIR = r"TASK - 4\leapGestRecog"  # 🔁 Change this
IMG_SIZE = (64, 64)

def load_dataset(path):
    X, y = [], []
    for subject in sorted(os.listdir(path)):
        subject_path = os.path.join(path, subject)
        if not os.path.isdir(subject_path): continue
        for gesture in sorted(os.listdir(subject_path)):
            gesture_path = os.path.join(subject_path, gesture)
            if not os.path.isdir(gesture_path): continue
            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    X.append(img)
                    y.append(gesture)
    return np.array(X), np.array(y)

# Load data
print("Loading images...")
X, y = load_dataset(DATASET_DIR)
X = X / 255.0  # Normalize
X = X.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.2f}")

# Save model
model.save("gesture_cnn_model.h5")

# Save label mapping
import pickle
with open("gesture_labels.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
