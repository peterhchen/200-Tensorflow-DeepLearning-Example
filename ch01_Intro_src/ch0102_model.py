import tensorflow as tf
#print ('tf.__version__: ', tf.__version__)

mnist = tf.keras.datasets.mnist # data set with 28x28 images of hand written digits 0-9
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize (x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1

model = tf.keras.models.Sequential()
# We want the flat model instead of 28x28 for the input layer
model.add(tf.keras.layers.Flatten()) 
# how many units (128 neurons) in the layer
# relu: rectifier linear for activation layer 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
# Add the final layer. The activation fucntion is probability distribution.
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) 

# Compile. Normal network does not attempt to optimize for accuracy.
# Choose popular adam optimizer, you can choose stochastic gradient descent
# loss: categorical cross entropy,
# Note: metrics mispell to "mertics" causes "TypeError: 'numpy.float64' object is not iterable"
model.compile (optimizer='adam',
         loss='sparse_categorical_crossentropy',
         metrics=['accuracy'])

# We are ready to train the model
model.fit (x_train, y_train, epochs=3) 

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

# validate the loss and accuracy
val_loss, val_acc = model.evaluate (x_test, y_test)
print ('val_loss: ', val_loss,'val_acc: ', val_acc)
predictions = new_model.predict (x_test)
print ('predictions: ', predictions)
import matplotlib.pyplot as plt
#print (x_train[0])
#plt.imshow(x_train[0])
plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.imshow(x_test[0])
plt.show()