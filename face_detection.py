import cv2
import face_recognition
import imutils
import numpy as np
import pickle
import sys

from datetime import datetime
from imutils.video import VideoStream, FPS
from os import path
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtGui


class RecordVideo(QtCore.QObject):
    """
    카메라의 동작을 활성화하고
    QTimer를 사용해 이미지 프레임을 포착하는 클래스
    """

    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)

        # PiCamera를 사용할 경우 30번 라인을 주석처리하고
        # 31번 라인의 코드의 주석을 해제
        self.camera = VideoStream(src=camera_port).start()
        #self.camera = VideoStream(usePiCamera=1).start()

        self.fps = FPS().start()
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        self.image_data.emit(self.camera.read())
    
    def __del__(self):
        """
        객체가 사라질 때, 즉 프로그램이 종료될 때
        소요된 시간과, 평균 fps 출력
        """
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))


class FaceDetectionWidget(QtWidgets.QWidget):
    """
    얼굴을 탐지, 식별, 인식 하는 클래스
    """
    def __init__(self, haar_cascade_filepath, detect, regist, parent=None):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.detect = detect
        self.regist = regist
        self.data = self.regist.data
        self.unknown = []
        self.unknown_name = 'Unknown'
        self.detected = False
        self.names = dict()
        for name in self.data['names']:
            self.names[name] = self.names.get(name, 0) + 1

    def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)
        faces = self.classifier.detectMultiScale(gray_image,
                                                scaleFactor=1.3,
                                                minNeighbors=4,
                                                flags=cv2.CASCADE_SCALE_IMAGE,
                                                minSize=self._min_size)
        return faces

    def image_data_slot(self, image_data):
        faces = self.detect_faces(image_data)
        rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(
                    self.data['encodings'],
                    encoding,
                    tolerance=0.5
                    )
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.data['names'][i]
                    counts[name] = counts.get(name, 0) + 1
                found_one = max(counts, key=counts.get)
                if counts[found_one] >= self.names[found_one] // 2:
                    name = found_one
                    self.detect.detect(name)
                else:
                    name = "Unknown"
                self.unknown_name = name
                self.unknown = encoding
                self.detected = True
            else:
                self.unknown_name = 'Unknown'
                self.unknown = encoding
                self.detected = True

            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image_data, (left, top), (right, bottom),
                    (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image_data, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        image = imutils.resize(image, width=400, height=480)
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                        QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def clear_unknown(self):
        self.unknown.clear()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    """
    PyQt 위젯들이 붙는 클래스
    """

    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)
        fp = haarcascade_filepath
        self.detect = Detect()
        self.regist = Regist()
        self.face_detection_widget = FaceDetectionWidget(fp, self.detect, self.regist)
        self.regist.face = self.face_detection_widget

        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        self.line_edit = QtWidgets.QLineEdit()
        layout.addWidget(self.run_button)
        layout.addWidget

        self.run_button.clicked.connect(self.record_video.start_recording)
        self.setFixedWidth(800)
        self.setFixedHeight(440)
        layout_base = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, self)
        self.setLayout(layout_base)
        grp_1 = QGroupBox("Camera")
        layout_base.addWidget(grp_1)
        grp_1.setLayout(layout)
        grp_2 = QGroupBox("Info")
        layout_base.addWidget(grp_2)
        layout2 = QtWidgets.QVBoxLayout()
        grp_2.setLayout(layout2)
        self.tbw = QTabWidget()
        layout2.addWidget(self.tbw)
        self.tbw.addTab(self.detect, Detect.__name__)
        self.tbw.addTab(self.regist, Regist.__name__)


class Regist(QWidget):
    """
    얼굴을 등록하는 tab을 구성하는 클래스
    """

    def __init__(self, parent=None):
        super(Regist, self).__init__(parent=parent)
        self.viewer = QListWidget(self)
        self.text = QLineEdit(self)
        self.button = QPushButton('등록')
        self.data = pickle.loads(open('encodings.pickle', 'rb').read())
        self.face = None
        self.names = set()

        self.init_widget()
        self.button.clicked.connect(self.regist)
        self.init_list()

    def init_widget(self):
        form_lbx = QGridLayout()
        self.setLayout(form_lbx)
        form_lbx.addWidget(self.text, 0, 0)
        form_lbx.addWidget(self.button, 1, 0)
        form_lbx.addWidget(self.viewer, 2, 0)
        form_lbx.setColumnStretch(0, 1)
        form_lbx.setColumnStretch(1, 3)
        self.text.setFixedWidth(300)
        self.button.setFixedWidth(300)
        self.viewer.setFixedWidth(300)

    def init_list(self):
        for name in self.data['names']:
            self.names.add(name)
        for name in self.names:
            self.viewer.addItem(name)

    def regist(self):
        if self.face.detected:
            new_name = self.text.text()
            if self.face.unknown_name != 'Unknown':
                if new_name != self.face.unknown_name:
                    print(new_name, self.face.unknown_name)
                    return
            self.data['encodings'].append(self.face.unknown)
            self.data['names'].append(new_name)
            self.face.names[new_name] = self.face.names.get(new_name, 0) + 1
            if not new_name in self.names:
                self.names.add(new_name)
                self.viewer.addItem(new_name)
        

class Detect(QWidget):
    """
    얼굴을 탐지, 식별하는 tab을 구성하는 클래스
    """

    def __init__(self, parent=None):
        super(Detect, self).__init__(parent=parent)
        self.viewer = QListWidget(self)
        self.init_widget()
        self.name_dict = dict()

    def init_widget(self):
        form_lbx = QGridLayout()
        self.setLayout(form_lbx)
        form_lbx.addWidget(self.viewer, 0, 0)
        form_lbx.setColumnStretch(0,1)
        form_lbx.setColumnStretch(1,3)
        self.viewer.setFixedWidth(300)

    def detect(self, name):
        now = datetime.now()
        if name in self.name_dict.keys():
            if (now-self.name_dict[name]).seconds < 10:
                return
        self.name_dict[name] = now
        time_text = '{0}/{1}/{2} {3}:{4}:{5} {6}'.format(
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            name
        )
        item = QListWidgetItem()
        item.setText(time_text)
        self.viewer.addItem(item)
        self.viewer.scrollToBottom()

def main(haar_cascade_filepath):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(haar_cascade_filepath)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = path.join(script_dir,
                                 'haarcascade_frontalface_default.xml')

    cascade_filepath = path.abspath(cascade_filepath)
    main(cascade_filepath)