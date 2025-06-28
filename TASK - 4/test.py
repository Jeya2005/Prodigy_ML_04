import cv2
import numpy as np
import tensorflow as tf
import pickle

# Load model and label encoder
model = tf.keras.models.load_model("gesture_cnn_model.h5")
with open("gesture_labels.pkl", "rb") as f:
    label_encoder = pickle.load(f)

IMG_SIZE = (64, 64)

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Define ROI (Region of Interest)
    roi = frame[100:300, 100:300]  # You can adjust this area
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    # Predict
    prediction = model.predict(reshaped)
    class_id = np.argmax(prediction)
    class_name = label_encoder.inverse_transform([class_id])[0]

    # Display prediction
    cv2.putText(frame, f'Gesture: {class_name}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
