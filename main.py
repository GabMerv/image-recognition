

import cv2
import numpy as np
from glob import glob
import random
from tqdm import tqdm
from keras.models import Sequential
from keras.utils import to_categorical
import h5py
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
import keras
import os


import random

x_train = []
x_test = []
y_train = [] # 1: des chats, 0: des chiens
y_test = []

size = (100, 100)

#! unzip *.zip && rm *.zip

print("Importation des chiens")
chiens = glob("<path: chiens>")
separation = len(chiens) // 5
i = 0
for image in tqdm(chiens):
   image = cv2.imread(image)
   image = cv2.resize(image, size)
   image = image.astype('float32')
   image /= 255
   if i < separation:
      x_test.append(image)
      y_test.append(0)
   else:
      x_train.append(image)
      y_train.append(0)
   i += 1

print("Importation des chats")
chats = glob("<path: chats>")
separation = len(chats) // 5
i = 0
for image in tqdm(chats):
   image = cv2.imread(image)
   image = cv2.resize(image, size)
   image = image.astype('float32')
   image /= 255
   if i < separation:
      x_test.append(image)
      y_test.append(1)
   else:
      x_train.append(image)
      y_train.append(1)
   i += 1


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)


brain = Sequential()

brain.add(Conv2D(256, kernel_size=(10,10), padding='same', input_shape=(100,100,3), activation='relu'))
brain.add(MaxPooling2D(pool_size=(2,2)))
brain.add(BatchNormalization())
brain.add(Dropout(0.4))

brain.add(Conv2D(64, kernel_size=(7,7), padding='same', activation='relu'))
brain.add(MaxPooling2D(pool_size=(2,2)))
brain.add(Dropout(0.3))

brain.add(LeakyReLU(alpha=0.1))
brain.add(Conv2D(32, kernel_size=(5,5), padding='same', activation='relu'))

brain.add(Conv2D(16, kernel_size=(4,4), padding='same', activation='relu'))

brain.add(Conv2D(16, kernel_size=(2,2), padding='same', activation='relu'))
brain.add(LeakyReLU(alpha=0.6))
brain.add(Flatten())

brain.add(Dense(128, activation='relu'))

brain.add(Dense(64, activation="relu"))

brain.add(Dense(32, activation="relu"))

brain.add(Dense(16, activation="relu"))

brain.add(Dense(16, activation="relu"))
brain.add(LeakyReLU(alpha=0.2))
brain.add(Dense(16, activation="relu"))
brain.add(LeakyReLU(alpha=0.1))
brain.add(Dense(2, activation="softmax"))


# callbacks = [
#    ModelCheckpoint("save_at_{epoch}.h5"),
# ]

brain.compile(loss="categorical_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])

#brain.fit(x_train, y_train,callbacks=callbacks,epochs=25, verbose=1, validation_data=(x_test,y_test), shuffle=True)

brain.fit(x_train, y_train,
   epochs=50,
   verbose=1,
   validation_data=(x_test,y_test),
   shuffle=True,
   max_queue_size=7,
   workers=2,
   use_multiprocessing = True,
   batch_size=8,
   )

brain.save("MIKEJOHNSON.h5")


