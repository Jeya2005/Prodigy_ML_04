import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# === Parameters ===
CAT_DIR = r"TASK - 3\Cat"
DOG_DIR = r"TASK - 3\Dog"
IMG_SIZE = 64
MAX_IMAGES = 4000
USE_PCA = True
PCA_COMPONENTS = 150

data = []
labels = []

# === Load Cat Images ===
for i, img_name in enumerate(tqdm(os.listdir(CAT_DIR))):
    if i >= MAX_IMAGES:
        break
    img_path = os.path.join(CAT_DIR, img_name)
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        data.append(img.flatten())
        labels.append(0)
    except:
        continue

# === Load Dog Images ===
for i, img_name in enumerate(tqdm(os.listdir(DOG_DIR))):
    if i >= MAX_IMAGES:
        break
    img_path = os.path.join(DOG_DIR, img_name)
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        data.append(img.flatten())
        labels.append(1)
    except:
        continue

# === Prepare Data ===
X = np.array(data)
y = np.array(labels)

# === Apply PCA (Optional) ===
if USE_PCA:
    print("Applying PCA...")
    pca = PCA(n_components=PCA_COMPONENTS)
    X = pca.fit_transform(X)

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Use GridSearchCV to Optimize SVM ===
print("Running GridSearchCV... This may take a few minutes.")
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']  # You can add 'poly' or 'sigmoid' too
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# === Evaluate ===
print("Best Parameters Found:", grid.best_params_)
y_pred = grid.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
