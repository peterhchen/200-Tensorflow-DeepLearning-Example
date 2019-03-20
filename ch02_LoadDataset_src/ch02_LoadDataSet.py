import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# numpy is do the array operation.
# matplotlib is used for plot.
# os is used for iterate the directory and join path 
# if you do no have cv2 use "pip install opencv-pthon" 
# cv2 is used to do image operation.

# Set the data directory
#DATADIR ="C:\Work\MicroService\Tensorflow\02_DeepLeaning_Tensoflow_Keara_sentdex_11\DataSets\PetImages"
DATADIR ="C:/Work/MicroService/Tensorflow/02_DeepLeaning_Tensoflow_Keara_sentdex_11/DataSets/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
for category in CATEGORIES:
    path = os.path.join (DATADIR, category) # path to cats or dogs
    for img in os.listdir (path):
        # If we want to read grayscale
        imag_array = cv2.imread(os.path.join (path, img), cv2.IMREAD_GRAYSCALE)
        # if we want to read RGB image.
        #imag_array = cv2.imread(os.path.join (path, img))
        new_array = cv2.resize (imag_array, (IMG_SIZE, IMG_SIZE))
        #plt.imshow(imag_array, cmap="gray")
        plt.imshow (new_array, cmap="gray")
        plt.show()
        break
    break
print ('imag_array.shep: ', imag_array.shape)
print ('image_array: '); print (imag_array)