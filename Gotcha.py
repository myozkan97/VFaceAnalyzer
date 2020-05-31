import cv2
import imutils
import numpy as np
import pafy
import models
import keras
import face_classifier_model
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import pathlib
import shutil

#@TODO #fix paths

# url of the video to predict Age and gender
url = 'https://www.youtube.com/watch?v=Psu9eE8ualg'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)


age_model = models.ageModel()
gender_model = models.genderModel()
emotion_model, emotion_labels = models.emotion_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
age_output_indexes = np.array([i for i in range(0, 101)])


def video_detector():
    frame = 0
    while True:
        frame += 1
        for i in range(30):
            cap.read()
        ret, image = cap.read()

        if ret is False:
            break

        image = imutils.resize(image, width=560)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            print("Found {} faces".format(str(len(faces))))
            for (x, y, w, h) in faces:
                if w > 50:
                    # age gender data set has 40% margin around the face. expand detected face.
                    margin = 30
                    margin_x = int((w * margin) / 100)
                    margin_y = int((h * margin) / 100)

                    detected_10margin_face = image[int(y):int(y + h), int(x):int(x + w)]

                    try:
                        detected_40margin_face = image[int(y - margin_y):int(y + h + margin_y), int(x - margin_x):int(x + w +
                                                                                                             margin_x)]
                        if detected_40margin_face.size == 0:
                            raise Exception()
                    except:
                        detected_40margin_face = detected_10margin_face


                    try:

                        detected_40margin_face = cv2.resize(detected_40margin_face, (224, 224))

                        detected_gray_face = cv2.resize(detected_10margin_face, (48, 48))
                        detected_gray_face = cv2.cvtColor(detected_gray_face, cv2.COLOR_BGR2GRAY)

                        img_pixels = keras.preprocessing.image.img_to_array(detected_40margin_face)
                        img_pixels = np.expand_dims(img_pixels, axis=0)
                        img_pixels /= 255

                        # Predict age and gender
                        age_dists = age_model.predict(img_pixels)
                        apparent_age = str(int(np.floor(np.sum(age_dists * age_output_indexes, axis=1))[0]))

                        gender_distribution = gender_model.predict(img_pixels)[0]
                        gender_index = np.argmax(gender_distribution)

                        detected_gray_face = keras.preprocessing.image.img_to_array(detected_gray_face)
                        detected_gray_face = np.expand_dims(detected_gray_face, axis=0)
                        detected_gray_face /= 255

                        emotion_prediction = emotion_labels[np.argmax(emotion_model.predict(detected_gray_face)[0])]

                        if gender_index == 0:
                            gender = "F"
                        else:
                            gender = "M"

                        # save picture to hard drive
                        save_picture(detected_10margin_face, frame, apparent_age, gender, emotion_prediction)

                        # Create an overlay text and put it into frame
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        overlay_text = "%s %s\n%s" % (gender, apparent_age, emotion_prediction)
                        cv2.putText(image, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    except Exception as e:
                        print("exception ", e)

            cv2.imshow('frame', image)
            # 0xFF is a hexadecimal constant which is 11111111 in binary.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def classify_and_folder_faces():
    path = r".\saved_images\\"

    #@TODO fix imports
    vgg_face = face_classifier_model.get_embeddings()

    # @TODO fix imports
    person_labels = {}
    with open("persons.json") as file:
        person_labels = json.loads(file.read())

    # @TODO fix imports
    classifier_model = tf.keras.models.load_model('face_classifier_model.h5')

    person_pics = os.listdir(path)

    for i, image_name in enumerate(person_pics):
        image = load_img(path + image_name, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        img_encode = vgg_face(image)

        embed = K.eval(img_encode)
        person = classifier_model.predict(embed)
        arg_max = np.argmax(person)
        if arg_max > 0.5:
            name = person_labels[str(arg_max)]
        else:
            continue

        if not os.path.isdir(path + name):
            os.mkdir(path + name)

        shutil.move(path + image_name, path + name + "\\" + image_name)


def analyze_classified_folders(folder_path):
    pass


def save_picture(picture, number, age, gender, emotion):
    saved_pictures_path = r".\saved_images\\"

    if not os.path.isdir(saved_pictures_path):
        os.mkdir(saved_pictures_path)

    cv2.imwrite(saved_pictures_path
                + str(number) + '_' + gender + '_' + age + '_' + emotion + ".jpg",
                picture)





if __name__ == "__main__":
    video_detector()
    classify_and_folder_faces()
