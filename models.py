import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
import json
import utils


def get_vgg_face_model() -> keras.models.Sequential:
    """This function returns vgg face model built according to the paper."""

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

    return model


def get_age_model() -> keras.models.Model:
    """This function returns age prediction model with it's pre-trained weights loaded."""

    model = get_vgg_face_model()

    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)

    file_path = utils.get_or_download("age_model_weights.h5", "https://drive.google.com/uc?id="
                                                              "1JRZTjxQbR07dNWm9_XhULMWJ-uGeZbZ8")

    age_model.load_weights(file_path)

    return age_model


def get_gender_model() -> keras.models.Model:
    """This function returns gender prediction model with it's pre-trained weights loaded."""
    model = get_vgg_face_model()

    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    file_path = utils.get_or_download("gender_model_weights.h5", "https://drive.google.com/uc?id="
                                                                 "1FDdW_U45pG2upMO3mKxsxQhb4YNIwEg2")

    gender_model.load_weights(file_path)

    return gender_model


def get_emotion_model() -> (keras.models.Model, list):
    """This function returns facial expression prediction model with it's pre-trained weights
    loaded and emotion labels which used to make sesse of the predictions."""

    model_str_file_path = utils.get_or_download("facial_expression_model_structure.json", "https://drive.google.com"
                                                                                          "/uc?id="
                                                                                          "1GzVBzxSuYfChqZP4efs6ZUl4"
                                                                                          "9Sn5G6Ws")
    model_weights_file_path = utils.get_or_download("facial_expression_model_weights.h5", "https://drive.google.com"
                                                                                          "/uc?id=1o5wjB5G1pfyY7ppPb"
                                                                                          "TIRIHMWRCoNUwkA")

    model = model_from_json(open(model_str_file_path, "r").read())
    model.load_weights(model_weights_file_path)

    labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    return model, labels


def get_vgg_face_model_embedding_extractor() -> keras.models.Model:
    """This function removes last two layers of vgg model and returns it. The resulting method
    is used to extract facial embeddings from a given 224x224 image. """
    # Define VGG_FACE_MODEL architecture
    model = get_vgg_face_model()

    file_path = utils.get_or_download('vgg_face_weights.h5', 'https://drive.google.com/uc?id='
                                                             '1Hut0a6bf3OpL_2kbgyqnYiu0nr_OXYTz')

    # Load VGG Face model weights
    model.load_weights(file_path)

    # Remove last Softmax layer and get model upto last flatten layer #with outputs 2622 units
    vgg_face_extractor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_extractor


def get_face_softmax_regressor_classifier_model_and_labels() -> (dict, keras.models.Model):
    """This function returns a dictionary which represents output results
     and our softmax regressor which classifies face images."""

    labels_file_path = utils.get_or_download('persons.json', 'https://drive.google.com/uc?id=1EDegJgHR76PCEyEFMn'
                                                             '-gbYJrO0v6Od6A')
    classifier_file_path = utils.get_or_download('face_classifier_model.h5', 'https://drive.google.com/uc?id='
                                                                             '1chpAcueZ2VSGb2QAF4HoowI9OJc_Sw5e')

    with open(labels_file_path) as fp:
        person_labels = json.load(fp)

    classifier_model = tf.keras.models.load_model(classifier_file_path)

    return person_labels, classifier_model

