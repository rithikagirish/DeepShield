import os
import numpy as np
import cv2
import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

REAL_DIR = "aISD/real_and_fake_face/training_real/"
FAKE_DIR = "aISD/real_and_fake_face/training_fake/"


def load_images(directory, label, img_size=(224, 224)):
    images, labels = [], []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img.astype("float32") / 255.0  
            images.append(img)
            labels.append(label)
    return np.array(images, dtype="float32"), np.array(labels, dtype="int32")  



real_images, real_labels = load_images(REAL_DIR, 0)
fake_images, fake_labels = load_images(FAKE_DIR, 1)


X = np.concatenate((real_images, fake_images), axis=0)
y = np.concatenate((real_labels, fake_labels), axis=0)

X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


base_model = efn.EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False


model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  
])


model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.save("deepfake_detector_3.h5")  
