import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import json
import os
import models


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
	person_folders = os.listdir('./faces/')
	for i, person in enumerate(person_folders):
		print("preparing train set for person: " + person)
		person_rep[i] = person
		image_names = os.listdir(r'.\faces\\' + person + '\\')
		for image_name in image_names:
			print("\tphoto: " + image_name)
			img = load_img('./faces/' + person + '/' + image_name, target_size=(224, 224))
			img = img_to_array(img)

			img = np.expand_dims(img, axis=0)
			img = preprocess_input(img)
			img_encode = vgg_face(img)
			x_train.append(np.squeeze(K.eval(img_encode)).tolist())
			y_train.append(i)

	return person_rep, np.array(x_train), np.array(y_train)


test_images_path = r'C:\Users\myozkan\PycharmProjects\aiproj2\test_images\\'

# create training set
person_rep, x_train, y_train = prepare_train_data(models.get_vgg_face_model_embedding_extractor())

# save training set
np.save("tain_data.npy", x_train)
np.save("train_labels.npy", y_train)

# save labels
with open('person_rep', 'w') as fp:
	json.dump(person_rep, fp)

# load saved training set
x_train = np.load("tain_data.npy")
y_train = np.load("train_labels.npy")

# laod labels
with open("person_rep") as fp:
	person_rep = json.load(fp)

# create and train softmax regressor
classifier_model = softmax_regressor(x_train, y_train)

# Save model for later use
tf.keras.models.save_model(classifier_model, './face_classifier_model2.h5')

# Load saved model
classifier_model = tf.keras.models.load_model(r'./face_classifier_model.h5')
