import cv2
import numpy as np
import pafy
import imutils
import models
import keras

# url of the video to predict Age and gender
url = 'https://www.youtube.com/watch?v=cZ-11eGOkKM'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


age_model = models.ageModel()
gender_model = models.genderModel()


#age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
output_indexes = np.array([i for i in range(0, 101)])

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return age_net, gender_net


def video_detector():
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        for i in range(25):
            cap.read()
        ret, image = cap.read()
        image = imutils.resize(image, width=1024)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if (len(faces) > 0):
            print("Found {} faces".format(str(len(faces))))
        for (x, y, w, h) in faces:
            if w > 20:


                # Get Face
                face_img = image[y:y + h, h:h + w].copy()


                # age gender data set has 40% margin around the face. expand detected face.
                margin = 30
                margin_x = int((w * margin) / 100);
                margin_y = int((h * margin) / 100)

                detected_face = image[int(y):int(y + h), int(x):int(x + w)].copy()


                try:
                    detected_face = image[int(y - margin_y):int(y + h + margin_y),
                                int(x - margin_x):int(x + w + margin_x)]
                    cv2.imshow("face_img", detected_face)
                except:
                    print("Detected face has not enough margin around to predict gender and age")


                try:
                    detected_face = cv2.resize(detected_face, (224, 224))

                    img_pixels = keras.preprocessing.image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255

                    # Predict age and gender
                    age_dists = age_model.predict(img_pixels)
                    apparent_age = str(int(np.floor(np.sum(age_dists * output_indexes, axis =1))[0]))

                    gender_distribution = gender_model.predict(img_pixels)[0]
                    gender_index = np.argmax(gender_distribution)

                    if gender_index == 0:
                        gender = "F"
                    else:
                        gender = "M"

                    # Create an overlay text and put it into frame
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    overlay_text = "%s %s" % (gender, apparent_age)
                    cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    print("exception ", e)

            cv2.imshow('frame', image)
            # 0xFF is a hexadecimal constant which is 11111111 in binary.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    video_detector()