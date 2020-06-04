from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import QThreadPool
from CustomUI import ImageLabel, Worker
import numpy as np
from tensorflow import keras
import Classifier
import cv2
import utils
import models
import imutils
import pafy

# path to save our images after finishing analysis of the video
saved_images_path = r".\saved_images\\"

# url of the video to predict Age and gender
url = 'https://www.youtube.com/watch?v=r-GFmH0EK9Y'


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Construct application
        self.windowClosed = False

        # Ust Bar
        video_url = QLineEdit()
        self.startButton = QPushButton("Analyze")
        header_box = QHBoxLayout()
        header_box.addWidget(QLabel("Youtube URL: "))
        header_box.addWidget(video_url)
        header_box.addWidget(self.startButton)

        # Log ve Resim
        self.textBox = QTextEdit()
        self.textBox.setReadOnly(True)
        self.textBox.setMinimumWidth(200)
        self.textBox.setMaximumWidth(600)
        self.imageBox = ImageLabel()
        self.imageBox.setMinimumWidth(400)
        content_box = QHBoxLayout()
        content_box.addWidget(self.textBox)
        content_box.addWidget(self.imageBox)

        self.threadpool = QThreadPool()

        # Events
        self.startButton.clicked.connect(self.analyze_click)

        # Butun pencereyi olustur
        layout = QVBoxLayout()
        layout.addLayout(header_box)
        layout.addLayout(content_box)

        widget = QWidget()
        widget.setLayout(layout)

        self.setWindowTitle("VFaceAnalyzr Alpha")
        self.resize(1280, 720)
        self.setCentralWidget(widget)

        self.show()
        self.setWindowIcon(QIcon('icon.png'))

    def analyze_video(self, progress_callback):
        """This function gets frames from a given video, crops faces and makes predictions out of them. After making
            predictions, it saves the frames to the hard drive with the help of "savePicture()" function and and shows
            them to the user with prediction labels. Cropped faces will be saved into the ./saved_images folder."""

        self.startButton.setDisabled(True)
        self.print_line("Starting real-time video analyzer...")

        v_pafy = pafy.new(url)
        play = v_pafy.getbest(preftype="mp4")
        cap = cv2.VideoCapture(play.url)

        # here we load our models to make out predictions
        age_model = models.get_age_model()
        gender_model = models.get_gender_model()
        emotion_model, emotion_labels = models.get_emotion_model()
        face_cascade = cv2.CascadeClassifier(
            utils.get_or_download('haarcascade_frontalface_default.xml', 'https://drive.google'
                                                                         '.com/uc?id=1vuWt_x_3'
                                                                         'QQaMs8nxklmMf-8OtHMB'
                                                                         'OM5V'))

        # age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
        age_output_indexes = np.array([i for i in range(0, 101)])

        utils.delete_contents_of_folder(saved_images_path)

        frame = 0
        frame_width = 720
        self.print_line("Started real-time video analyzer...")
        while not self.windowClosed:
            frame += 1
            for i in range(100):
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
                    if w > frame_width / 10:
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
                            Classifier.save_picture(detected_10margin_face, frame, apparent_age, gender,
                                                    emotion_prediction)

                            # Create an overlay text and put it into frame
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                            overlay_text = "%s %s %s" % (gender, apparent_age, emotion_prediction)
                            cv2.putText(image, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                        except Exception as e:
                            print("exception ", e)

                self.show_image(image)
                # 0xFF is a hexadecimal constant which is 11111111 in binary.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.print_line("Classifying saved images, please wait...")
        Classifier.classify_and_folder_faces()

        self.print_line("Generating report, please wait...")
        total_number_of_images, person_dictionary = Classifier.analyze_classified_folders()

        self.print_line(Classifier.create_report(total_number_of_images, person_dictionary))

    def show_image(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.imageBox.pixmap = QPixmap.fromImage(out_image)
        self.imageBox.repaint()

    def print_line(self, text):
        self.textBox.insertPlainText(text + "\n")

    def clean_textbox(self):
        self.textBox.clear()

    def analyze_complete(self):
        self.startButton.setDisabled(False)

    def analyze_click(self):
        self.clean_textbox()
        worker = Worker(self.analyze_video)
        worker.signals.finished.connect(self.analyze_complete)
        self.threadpool.start(worker)

    def closing_window(self):
        self.window_closed = True


app = QApplication([])
window = MainWindow()
app.exec_()
