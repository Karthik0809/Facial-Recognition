"""Training script for the facial recognition CNN model.

This script downloads the LFW dataset, trains a convolutional neural
network with data augmentation and stores the best model weights to
``model.h5``. The saved model can be loaded by ``fastapi_app.py`` for
serving predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Dropout,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import fetch_lfw_people


def load_data():
    """Load the LFW dataset and split it into train and test sets."""
    faces = fetch_lfw_people(min_faces_per_person=100)
    image_count, h, w = faces.images.shape[:3]
    class_count = len(faces.target_names)

    mask = np.zeros(faces.target.shape, dtype=bool)
    for target in np.unique(faces.target):
        mask[np.where(faces.target == target)[0][:150]] = True

    x_faces = faces.images[mask].reshape((-1, h, w, 1)) / 255.0
    y_faces = to_categorical(faces.target[mask])

    x_train, x_test, y_train, y_test = train_test_split(
        x_faces, y_faces, train_size=0.8, stratify=faces.target[mask], random_state=0
    )
    return (x_train, x_test, y_train, y_test, faces.target_names, h, w, class_count)


def build_model(input_shape, num_classes):
    """Return a deeper CNN model."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train():
    x_train, x_test, y_train, y_test, names, h, w, class_count = load_data()

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    model = build_model((h, w, 1), class_count)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint("model.h5", save_best_only=True),
    ]

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        validation_data=(x_test, y_test),
        epochs=50,
        callbacks=callbacks,
    )

    # Evaluation
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=names))

    # Confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return history


if __name__ == "__main__":
    train()
