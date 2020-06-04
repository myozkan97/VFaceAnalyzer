import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import shutil
import models

# path to save our images after finishing analysis of the video
saved_images_path = r".\saved_images\\"


def classify_and_folder_faces() -> None:
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
        img_encode = vgg_face(image)

        embed = K.eval(img_encode)
        person = classifier_model.predict(embed)
        arg_max = np.argmax(person)
        if arg_max > 0.8:
            name = person_labels[str(arg_max)]
        else:
            name = "unknown"

        if not os.path.isdir(saved_images_path + name):
            os.mkdir(saved_images_path + name)

        shutil.move(saved_images_path + image_name, saved_images_path + name + "\\" + image_name)


def analyze_classified_folders() -> (int, dict):
    """This function analyzes folder-classified pictures and created a dictionary which contains data about
    characters: All recorded age, gender, emotion predictions; name of the character etc... """

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


def save_picture(picture: np.array, number: int, age: str, gender: str, emotion: str) -> None:
    """This function saves given image and names saved file with given attributes."""
    saved_pictures_path = r".\saved_images\\"

    if not os.path.isdir(saved_pictures_path):
        os.mkdir(saved_pictures_path)

    cv2.imwrite(saved_pictures_path
                + str(number) + '_' + gender + '_' + age + '_' + emotion + ".jpg",
                picture)


def create_report(num_of_pictures: int, person_dict: dict) -> str:
    """This fuction is used to generate report string from a given dictionary input."""

    person_info_strs = []
    num_of_known_faces = 0

    screen_share_str = "Some people stole the show! Here are the screen share percentage of characters!:\n"

    for person in person_dict.keys():
        num_of_entries = len(person_dict[person]["age"])

        if person != "unknown" and person != "notface":
            screen_share_str += "\t%s has the screen share of: %.2f\n" % \
                                    (person.upper(), num_of_entries / num_of_pictures * 100)
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

    return final_str
