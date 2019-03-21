import tensorflow as tf
#print ('tf.__version__: ', tf.__version__)

mnist = tf.keras.datasets.mnist # data set with 28x28 images of hand written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
#print (x_train[0])
#plt.imshow(x_train[0])
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()