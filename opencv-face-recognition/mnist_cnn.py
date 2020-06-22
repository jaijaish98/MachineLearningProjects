# mnist_cnn.py
# uses built-in keras.datasets data

# ==================================================================

import numpy as np
import keras as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt

def main():
  # 0. get started
  print("\nMNIST image recognition using Keras/TensorFlow ")
  np.random.seed(1)
  tf.compat.v1.set_random_seed(1)

  # 1. load data
  print("Loading Keras version MNIST data into memory \n")
  (train_x,train_y),(test_x,test_y) = K.datasets.mnist.load_data()
  train_x = train_x.reshape(60000, 28, 28, 1).astype(np.float32)
  test_x = test_x.reshape(10000, 28, 28, 1).astype(np.float32)
  train_x /= 255; test_x /= 255

  train_y = K.utils.to_categorical(train_y, 10).astype(np.float32)
  test_y = K.utils.to_categorical(test_y, 10).astype(np.float32)

  # 2. define model
  init = K.initializers.glorot_uniform(seed=1)
  model = K.models.Sequential()
  model.add(K.layers.Conv2D(filters=32, kernel_size=(3,3),
    strides=(1,1), padding='same', kernel_initializer=init,
    activation='relu', input_shape=(28,28,1)))
  model.add(K.layers.Conv2D(filters=64, kernel_size=(3,3),
    strides=(1,1), padding='same', kernel_initializer=init,
    activation='relu'))
  model.add(K.layers.MaxPooling2D(pool_size=(2,2)))
  model.add(K.layers.Dropout(0.25))
  model.add(K.layers.Flatten())
  model.add(K.layers.Dense(units=100, kernel_initializer=init,
    activation='relu'))
  model.add(K.layers.Dropout(0.5))
  model.add(K.layers.Dense(units=10, kernel_initializer=init,
    activation='softmax'))

  model.compile(loss='categorical_crossentropy',
   optimizer='adadelta', metrics=['acc'])

  # 3. train model
  bat_size = 128
  max_epochs = 3  # too few

  print("Starting training ")
  model.fit(train_x, train_y, batch_size=bat_size,
    epochs=max_epochs, verbose=1)
  print("Training complete")

  # 4. evaluate model
  loss_acc = model.evaluate(test_x, test_y, verbose=0)
  print("\nTest data loss = %0.4f  accuracy = %0.2f%%" % \
(loss_acc[0], loss_acc[1]*100) )

  # 5. save model
  print("Saving model to disk \n")
  mp = ".\mnist_model.h5"
  model.save(mp)

  # 6. use model
  print("Using model to predict dummy digit image: ")
  unknown = np.zeros(shape=(28,28), dtype=np.float32)
  for row in range(5,23): unknown[row][9] = 180  # vertical line
  for rc in range(9,19): unknown[rc][rc] = 250   # diagonal line
  plt.imshow(unknown, cmap=plt.get_cmap('gray_r'))
  plt.show()

  unknown = unknown.reshape(1, 28,28,1)
  predicted = model.predict(unknown)
  print("\nPredicted digit is: ")
  print(predicted)

  # lbls = ["zero", "one", "two", "three", "four", "five", 
  #   "six", "seven", "eight", "nine"]
  # idx = np.argmax(predicted[0])
  # print(lbls[idx])

# ==================================================================

if __name__=="__main__":
  main()
