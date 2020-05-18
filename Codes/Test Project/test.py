from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
import pandas as panda
import numpy
import cv2 as cv
import os
from directories import TRAIN_DIR, TEST_DIR, TEST_LABELS_DIR, TRAIN_LABELS_DIR

training_data_gen = ImageDataGenerator(rescale=1./255)
training_data_gen = ImageDataGenerator(rescale=1./255)
train_generator = training_data_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(1024, 1024),
    color_mode='grayscale',
    class_mode='categorical'
)

validation_data_gen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_data_gen.flow_from_directory(
    TEST_DIR,
    target_size=(1024, 1024),
    color_mode='grayscale',
    class_mode='categorical'
)


model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1024, 1024, 1)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(
    train_generator,
    epochs=25,
    steps_per_epoch=20,
    validation_data=validation_generator,
    verbose=1,
    validation_steps=3
)
