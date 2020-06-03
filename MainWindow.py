from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtCore import Qt, QPoint, QThreadPool
from CustomUI import ImageLabel, Worker
import sys
import Gotcha

# Construct application
app = QApplication([])
app.setStyle('Fusion')
windowClosed = False

# Ust Bar
videoURL = QLineEdit()
startButton = QPushButton("Analyze")
headerBox = QHBoxLayout()
headerBox.addWidget(QLabel("Youtube URL: "))
headerBox.addWidget(videoURL)
headerBox.addWidget(startButton)

# Log ve Resim
textBox = QTextEdit()
textBox.setReadOnly(True)
textBox.setMinimumWidth(200)
textBox.setMaximumWidth(600)
imageBox = ImageLabel()
imageBox.setMinimumWidth(400)
contentBox = QHBoxLayout()
contentBox.addWidget(textBox)
contentBox.addWidget(imageBox)
threadpool = QThreadPool()


def analyze_video(progress_callback):
    startButton.setDisabled(True)
    # url of the video to predict Age and gender
    Gotcha.url = videoURL.text()

    Gotcha.main()


def show_image(img):
    qformat = QImage.Format_Indexed8
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888
    out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
    out_image = out_image.rgbSwapped()
    imageBox.pixmap = QPixmap.fromImage(out_image)
    imageBox.repaint()


def print_line(text):
    textBox.insertPlainText(text + "\n")


def clean_textbox():
    textBox.clear()

def analyze_complete():
    startButton.setDisabled(False)


def analyze_click():
    worker = Worker(analyze_video)
    worker.signals.finished.connect(analyze_complete)
    threadpool.start(worker)


def closing_window():
    window_closed = True

# Events
startButton.clicked.connect(analyze_click)

# Butun pencereyi olustur
layout = QVBoxLayout()
layout.addLayout(headerBox)
layout.addLayout(contentBox)

# Set Window
window = QWidget()
window.setWindowTitle("VFaceAnalyzr Alpha")
window.resize(640, 480)
window.setLayout(layout)
window.show()
app.aboutToQuit.connect(closing_window)


# Execute application
sys.exit(app.exec_())
threadpool.waitForDone(1000)