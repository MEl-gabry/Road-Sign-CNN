import fnmatch
from math import ceil
import xml.etree.ElementTree as ET
import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, InputLayer, BatchNormalization
from sklearn.model_selection import train_test_split

count = len(fnmatch.filter(os.listdir(os.path.join(os.path.dirname(__file__), 'images')), '*.*'))

X = []
y = []
X_train = []
X_test = []
y_train = []
y_test = []

bounding_boxes = []

categories = {}

width = 75
height = 75

for i in range(count):
    im = Image.open(f"images/road{i}.png")
    im = im.resize((width, height))
    im_array = np.array(im)

    tree = ET.parse(f"annotations/road{i}.xml")
    root = tree.getroot()
    size = root.find("size")
    im_width = int(size.find("width").text)
    im_height = int(size.find("height").text)
    obj = root.find("object")
    category = obj.find("name").text
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    bounding_boxes.append([xmin / im_width, ymin / im_height, xmax / im_width, ymax / im_height])

    if categories.get(category) is None:
        categories[category] = len(categories)

    cat_index = categories[category]

    X.append(im_array)
    y.append(cat_index)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = X_train / 255
X_test = X_test / 255

Y_train = to_categorical(y_train, len(categories))
Y_test = to_categorical(y_test, len(categories))

# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(width, height, 4)))
# add: max pool layer
model.add(MaxPool2D(pool_size=(2,2)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# add: batch normalization
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(4, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_data=(X_test, Y_test))