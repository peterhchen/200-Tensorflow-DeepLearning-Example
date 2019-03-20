import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR ="C:/Work/MicroService/Tensorflow/02_DeepLeaning_Tensoflow_Keara_sentdex_11/DataSets/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
training_data = []
def create_training_data ():
    for category in CATEGORIES:
        path = os.path.join (DATADIR, category) # path to cats or dogs
        class_num = CATEGORIES.index(category)  # category = 0 for dog, 1 for cat
        for img in os.listdir (path):
            try:
                imag_array = cv2.imread(os.path.join (path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize (imag_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass    # some of the images have reading error, just pass the error

create_training_data()

print (len(training_data))

import random
random.shuffle (training_data)
# Taks first 10 set of data.
for sample in training_data[:2]:
    # check the label is correct, sample [0]: image data, sample[1]: 0 (dog), 1 (cat)
    print (sample)

X = []    # features
y = []    # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

# X has to be a numpy array, -1 is how many features we going to store at index -1.
# image size ix IMG_SIZE. The last 1 is grayscale. 
# If the last one is 3, then RBG value
X = np.array(X).reshape (-1, IMG_SIZE, IMG_SIZE, 1)

import pickle
pickle_out = open ("X.pickle", "wb")
pickle.dump (X, pickle_out)
pickle_out.close()

pickle_out = open ("y.pickle", "wb")
pickle.dump (y, pickle_out)
pickle_out.close()

pickle_in = open ("x.pickle", "rb")
X = pickle.load (pickle_in)

pickle_in = open ("y.pickle", "rb")
y = pickle.load (pickle_in)

# Those are the data we setup for X
print ('X[:2]: ', X[:2])
# Those are the data we setup for y.
print ('y[:2]: ', y[:2])