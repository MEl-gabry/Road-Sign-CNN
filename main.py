import fnmatch
from math import ceil
import xml.etree.ElementTree as ET
import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical

count = len(fnmatch.filter(os.listdir(os.path.join(os.path.dirname(__file__), 'images')), '*.*'))

X_train = []
X_test = []
y_train = []
y_test = []

bounding_boxes = []

categories = {}

width = 200
height = 200

for i in range(count):
    im = Image.open(f"images/road{i}.png")
    im = im.resize((width, height))
    im_array = np.array(im)

    tree = ET.parse(f"annotations/road{i}.xml")
    root = tree.getroot()
    obj = root.find("object")
    category = obj.find("name").text
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    bounding_boxes.append([xmin / width, ymin / height, xmax / width, ymax / height])

    if categories.get(category) is None:
        categories[category] = len(categories)

    cat_index = categories[category]

    if i < ceil(count / 2):
        X_train.append(im_array)
        y_train.append(cat_index)
    else:
        X_test.append(im_array)
        y_test.append(cat_index)

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = X_train / 255
X_test = X_test / 255

Y_train = to_categorical(y_train, len(categories))
Y_test = to_categorical(y_test, len(categories))