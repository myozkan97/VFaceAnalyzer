import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import json
import os
import glob


path=r'C:\Users\myozkan\PycharmProjects\aiproj2'


def get_embeddings():
	# Define VGG_FACE_MODEL architecture
	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))

	# Load VGG Face model weights
	model.load_weights('vgg_face_weights.h5')

	# Remove last Softmax layer and get model upto last flatten layer #with outputs 2622 units
	vgg_face = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

	return vgg_face


def softmax_regressor(x_train, y_train):
	classifier_model = Sequential()
	classifier_model.add(Dense(units=200, input_dim=x_train.shape[1], kernel_initializer='glorot_uniform'))
	classifier_model.add(BatchNormalization())
	classifier_model.add(Activation('tanh'))
	classifier_model.add(Dropout(0.3))
	classifier_model.add(Dense(units=30, kernel_initializer='glorot_uniform'))
	classifier_model.add(BatchNormalization())
	classifier_model.add(Activation('tanh'))
	classifier_model.add(Dropout(0.2))
	classifier_model.add(Dense(units=21, kernel_initializer='he_uniform'))
	classifier_model.add(Activation('softmax'))
	classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='nadam', metrics=['accuracy'])

	print("fitting model with dataset x")
	classifier_model.fit(x_train, y_train, epochs=100)

	return classifier_model


def prepare_train_data(vgg_face):
	# Prepare Train Data
	x_train = []
	y_train = []
	person_rep = dict()
	person_folders = os.listdir(path + '/faces/')
	for i, person in enumerate(person_folders):
		print("preparing train set for person: " + person)
		person_rep[i] = person
		image_names = os.listdir(path + '\\' + 'faces\\' + person + '\\')
		for image_name in image_names:
			print("\tphoto: " + image_name)
			img = load_img(path + '/faces/' + person + '/' + image_name, target_size=(224, 224))
			img = img_to_array(img)

			img = np.expand_dims(img, axis=0)
			img = preprocess_input(img)
			img_encode = vgg_face(img)
			x_train.append(np.squeeze(K.eval(img_encode)).tolist())
			y_train.append(i)

	return person_rep, np.array(x_train), np.array(y_train)


test_images_path = r'C:\Users\myozkan\PycharmProjects\aiproj2\test_images\\'



vgg_face = get_embeddings()


x_train = np.load("train_data.npy")
y_train = np.load("train_labels.npy")


person_labels = {}
with open('persons.json') as fp:
	person_labels = json.load(fp)

# classifier_model = softmax_regressor(x_train, y_train)
#
# # Save model for later use
# tf.keras.models.save_model(classifier_model, path + '/face_classifier_model.h5')

# Load saved model
classifier_model = tf.keras.models.load_model(path + r'/face_classifier_model.h5')

