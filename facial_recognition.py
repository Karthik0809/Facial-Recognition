import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import fetch_lfw_people
import random

"""# Load Dataset"""

faces = fetch_lfw_people(min_faces_per_person=100)
image_count, image_height, image_width = faces.images.shape[:3]
class_count = len(faces.target_names)
print(f"Dataset contains {image_count} images of size {image_height}x{image_width} across {class_count} individuals.")
print("People in the dataset:", faces.target_names)

"""# Display Sample Images"""

fig, axes = plt.subplots(3, 8, figsize=(18, 10))
for idx, ax in enumerate(axes.flat):
    if idx < image_count:
        ax.imshow(faces.images[idx], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(faces.target_names[faces.target[idx]])
    else:
        ax.axis('off')
plt.show()

unique_targets, counts = np.unique(faces.target, return_counts=True)

for person, count in zip(faces.target_names[unique_targets], counts):
    print(f"{person}: {count} images")

"""# Data Preprocessing"""

mask = np.zeros(faces.target.shape, dtype=np.bool_)

for target in np.unique(faces.target):
    mask[np.where(faces.target == target)[0][:150]] = 1

x_faces = faces.data[mask]
y_faces = faces.target[mask]
x_faces.shape

face_images = x_faces / 255
face_labels = to_categorical(y_faces)

# ``train_test_split`` expects the ``stratify`` argument to be a 1-D array of
# class labels. Using the one-hot encoded labels caused a ``ValueError`` at
# runtime. The integer encoded ``y_faces`` provides the correct label vector
# for stratification.
x_train, x_test, y_train, y_test = train_test_split(
    face_images,
    face_labels,
    train_size=0.8,
    stratify=y_faces,
    random_state=0,
)

"""# CNN Model"""

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(image_width * image_height,)))
model.add(Dense(class_count, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""# Train Model"""

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=20)

"""# Plot Training History"""

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
plot_history(history)

"""# Model Evaluation"""

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=faces.target_names))

"""# Confusion Matrix"""

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred_classes), annot=True, fmt='d', cmap='Blues', xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""# Random Image Prediction"""

import random
import matplotlib.pyplot as plt

def predict_random_image():
    idx = random.randint(0, len(x_test) - 1)
    img = x_test[idx]
    true_label = faces.target_names[y_true[idx]]
    pred_label = faces.target_names[y_pred_classes[idx]]
    img = img.reshape(62, 47)

    plt.imshow(img, cmap='gray')
    plt.title(f'True: {true_label}\nPredicted: {pred_label}')
    plt.axis('off')
    plt.show()

predict_random_image()

