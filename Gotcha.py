import cv2
import imutils
import pafy
import models
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import shutil

# @TODO #fix paths

# path to save our images after finishing analysis of the video
saved_images_path = r".\saved_images\\"


# url of the video to predict Age and gender
url = 'https://www.youtube.com/watch?v=r-GFmH0EK9Y'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)


# here we load our models to make out predictions
age_model = models.get_age_model()
gender_model = models.get_gender_model()
emotion_model, emotion_labels = models.get_emotion_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
age_output_indexes = np.array([i for i in range(0, 101)])


def video_detector():
    """This function gets frames from a given video, crops faces and makes predictions out of them. After making
    predictions, it saves the frames to the hard drive with the help of "savePicture()" function and and shows them
    to the user with prediction labels. Cropped faces will be saved into the ./saved_images folder."""

    frame = 0
    frame_width = 960
    while True:
        frame += 1
        for i in range(30):
            cap.read()
        ret, image = cap.read()

        if ret is False:
            break

        image = imutils.resize(image, frame_width)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            print("Found {} faces".format(str(len(faces))))
            for (x, y, w, h) in faces:
                if w > frame_width / 15:
                    # age gender data set has 40% margin around the face. expand detected face.
                    margin = 30
                    margin_x = int((w * margin) / 100)
                    margin_y = int((h * margin) / 100)

                    detected_10margin_face = image[int(y):int(y + h), int(x):int(x + w)]

                    try:
                        detected_40margin_face = \
                            image[int(y - margin_y):int(y + h + margin_y), int(x - margin_x): int(x + w + margin_x)]

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
                        cv2.putText(image, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)
                    except Exception as e:
                        print("exception ", e)

            cv2.imshow('frame', image)
            # 0xFF is a hexadecimal constant which is 11111111 in binary.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def classify_and_folder_faces():
    """After finishing real time age, gender and emotion analysis of a given video, this function classifies saved
    face images with our hand trained face classifier model. In short, it saves all faces belonging to same person
    into same folder. For example, all faces belonging to Pam character, will be found in the ./saved_images/Pam
    folder. """

    vgg_face = models.get_vgg_face_model_embedding_extractor()

    person_labels, classifier_model = models.get_face_softmax_regressor_classifier_model_and_labels()

    person_pics = os.listdir(saved_images_path)

    for i, image_name in enumerate(person_pics):
        image = load_img(saved_images_path + image_name, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        print("====================== ", type(image), "===========================")
        img_encode = vgg_face(image)

        embed = K.eval(img_encode)
        person = classifier_model.predict(embed)
        arg_max = np.argmax(person)
        if arg_max > 0.5:
            name = person_labels[str(arg_max)]
        else:
            continue

        if not os.path.isdir(saved_images_path + name):
            os.mkdir(saved_images_path + name)

        shutil.move(saved_images_path + image_name, saved_images_path + name + "\\" + image_name)


def analyze_classified_folders():
    person_dict = {}

    if not os.path.exists(saved_images_path):
        os.mkdir(saved_images_path)

    person_folders = os.listdir(saved_images_path)

    total_num_of_images = 0
    for i, folder_name in enumerate(person_folders):
        person_dict[folder_name] = {key: [] for key in ["age", "gender", "emotion"]}
        image_names = os.listdir(saved_images_path + folder_name)
        for k, picture_name in enumerate(image_names):
            num, gender, age, emotion = picture_name.split("_")
            person_dict[folder_name]["gender"].append(gender)
            person_dict[folder_name]["age"].append(int(age))
            person_dict[folder_name]["emotion"].append(emotion[:-4])
            total_num_of_images += 1

    return total_num_of_images, person_dict


def save_picture(picture, number, age, gender, emotion):
    saved_pictures_path = r".\saved_images\\"

    if not os.path.isdir(saved_pictures_path):
        os.mkdir(saved_pictures_path)

    cv2.imwrite(saved_pictures_path
                + str(number) + '_' + gender + '_' + age + '_' + emotion + ".jpg",
                picture)


def create_report(num_of_pictures, person_dict):
    person_info_strs = []
    num_of_known_faces = 0

    screen_share_str = "Some people stole the show! Here are the screen share percentage of characters!:\n"

    for person in person_dict.keys():
        num_of_entries = len(person_dict[person]["age"])

        if person != "unknown" and person != "notface":
            screen_share_str += "\t%s has the screen share of: %.2f\n" % (
            person.upper(), num_of_entries / num_of_pictures * 100)
        elif person == "unknown":
            screen_share_str += "Unknown character/s have the screen share of: %.2f\n" % \
                                (num_of_entries / num_of_pictures * 100)

        if person != "unknown" and person != "notface":
            avg_age = 0
            for age in person_dict[person]["age"]:
                avg_age += age
            avg_age /= num_of_entries

            gender_predictions = {"F": 0, "M": 0}
            for gender in person_dict[person]["gender"]:
                gender_predictions[gender] += 1

            dominant_gender = "Female" if gender_predictions["F"] > gender_predictions["M"] \
                else "Male"

            emotion_rates = {key: 0 for key in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
            for emotion in person_dict[person]["emotion"]:
                emotion_rates[emotion] += 1

            tmp_str = "Character: %s: \nDominant gender prediction: %s.\nEstimated average age: %d\n" % \
                      (person.upper(), dominant_gender, int(avg_age))

            tmp_str += "Throughout the video, observed emotions are:\n "
            for emotion in emotion_rates.keys():
                tmp_str += "\t%s: %.2f\n" % (emotion.upper(), emotion_rates[emotion] / num_of_entries * 100)

            person_info_strs.append(tmp_str)
            num_of_known_faces += 1

    final_str = "In total %s images found by the face detector algorithm. \n\n" \
                "%s\n\n" \
                "There were %s known faces recognized by our Softmax Regressor. \n%s\n\n" % \
                (num_of_pictures, screen_share_str, num_of_known_faces, "\n\n".join(person_info_strs))

    print(final_str)


if __name__ == "__main__":
    #video_detector()
    classify_and_folder_faces()
    total_number_of_images, person_dictionary = analyze_classified_folders()
    create_report(total_number_of_images, person_dictionary)
