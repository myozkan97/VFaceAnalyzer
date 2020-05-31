import cv2
import imutils
import pafy

url = 'https://www.youtube.com/watch?v=o1yUKfvnqRQ'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture("office_s07e24.mp4")





def video_detector():
    font = cv2.FONT_HERSHEY_SIMPLEX
    k = 0
    while True:
        k += 1
        for i in range(20):
            cap.read()
        ret, image = cap.read()
        image = imutils.resize(image, width=480)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            print("Found {} faces".format(str(len(faces))))
        for (x, y, w, h) in faces:
            if w > 60:

                # age gender data set has 40% margin around the face. expand detected face.
                margin = 30
                margin_x = int((w * margin) / 100)
                margin_y = int((h * margin) / 100)

                detected_face = image[int(y):int(y + h), int(x):int(x + w)].copy()

                try:
                    detected_face = cv2.resize(detected_face, (224, 224))
                    print(cv2.imwrite(r'C:\Users\myozkan\PycharmProjects\aiproj2\test_images'
                                      r'\\' + str(k) + ".jpg",
                                      detected_face))



                    # Create an overlay text and put it into frame
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                except Exception as e:
                    print("exception ", e)

            cv2.imshow('frame', image)
            # 0xFF is a hexadecimal constant which is 11111111 in binary.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    video_detector()
