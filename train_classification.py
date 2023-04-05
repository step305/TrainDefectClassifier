import os
import random
import tkinter
from tkinter import filedialog as file_dlg

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths as imutils_paths
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from backend.lenet import LeNet

EPOCHS = 200
INIT_LR = 1e-4
BS = 32
IMAGE_SIZE = (64, 64)
N = 0
LOSS_HISTORY = []
VAL_LOSS_HISTORY = []
fig = None


class LogProcessCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print('Train\'ve started...')
        pass

    def on_epoch_end(self, epoch, logs=None):
        print('loss = {:.4f}\taccuracy = {:.4f}'.format(logs['loss'], logs['accuracy']))


if __name__ == '__main__':
    matplotlib.use("Agg")
    root = tkinter.Tk()
    dir_path = file_dlg.askdirectory(mustexist=True)
    root.update()
    root.destroy()
    print(dir_path)
    dataset_path = sorted(list(imutils_paths.list_images(dir_path)))

    print("[INFO] loading images...")
    data = []
    labels = []

    for imagePath in dataset_path:
        image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, IMAGE_SIZE)
        image = img_to_array(image)
        data.append(image)

        label = int(imagePath.split(os.path.sep)[-2].split('_')[-1]) - 1
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    num_of_unique_classes = len(set(labels))

    random.seed(30542)
    random.shuffle(dataset_path)

    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.25, random_state=30542)

    trainY = to_categorical(trainY, num_classes=num_of_unique_classes)
    testY = to_categorical(testY, num_classes=num_of_unique_classes)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    print("[INFO] compiling model...")

    model = LeNet.build(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], depth=3, classes=num_of_unique_classes)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    print("[INFO] training network...")
    H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
                  validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                  epochs=EPOCHS, verbose=0,
                  callbacks=[LogProcessCallback()]
                  )

    print("[INFO] serializing network...")
    model.save('classification_model.h5', save_format="h5")
