# USAGE
# python train_face_detector_model.py --dataset dataset

# import the necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import *
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
                default="face_detector.model",
                help="path to output face detector model")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 2
BS = 32

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, color_mode = "grayscale", target_size=(28, 28))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays
print(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)
# perform one-hot encoding on the labels
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainImages, testImages, trainLabels, testLabels) = train_test_split(data, labels,
                                                                      test_size=0.20, stratify=labels, random_state=42)

# fashion_mnist = keras.datasets.fashion_mnist
# (trainImages, trainLabels), (testImages, testLabels) = fashion_mnist.load_data()
# print(trainImages.shape, trainLabels.shape, testImages.shape, testLabels.shape)
# construct the training image generator for data augmentation
# aug = ImageDataGenerator(
# 	rotation_range=20,
# 	zoom_range=0.15,
# 	width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	shear_range=0.15,
# 	horizontal_flip=True,
# 	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
# baseModel = MobileNetV2(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
# model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
# for layer in baseModel.layers:
# 	layer.trainable = False

# compile our model
# print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="binary_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])

# train the head of the network
# print("[INFO] training head...")
# H = model.fit(
# 	aug.flow(trainImages, trainLabels, batch_size=BS),
# 	steps_per_epoch=len(trainImages) // BS,
# 	validation_data=(testImages, testLabels),
# 	validation_steps=len(testImages) // BS,
# 	epochs=EPOCHS)

# make predictions on the testing set
# print("[INFO] evaluating network...")
# predIdxs = model.predict(testImages, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
# predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
# print(classification_report(testLabels.argmax(axis=0), predIdxs,
# 	target_names=encoder.classes_))
trainImages = trainImages / 255.0
testImages = testImages / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(trainImages, trainLabels, epochs=10)
test_loss, test_acc = model.evaluate(testImages,  testLabels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

testImage = load_img("examples\\0004.jpg", color_mode="grayscale", target_size=(28, 28))
testImage = img_to_array(testImage)
testImage = preprocess_input(testImage)
l = [testImage]
l = np.array(l)

predictions = probability_model.predict(l)
print(predictions[0])
print(np.argmax(predictions[0]))


# serialize the model to disk
# print("[INFO] saving face detector model...")
# model.save(args["model"], save_format="h5")

# plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])
