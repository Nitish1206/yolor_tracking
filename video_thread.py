
'''cv2 video capture Thread'''
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import cv2

import time
import traceback


class videoThread(QThread):

    '''Custom SIGNALS'''
    # Image Signal
    image_signal = pyqtSignal(QImage)
    # Signal emitted when training has been completed
    # training_done_signal = pyqtSignal(bool)
    # Signal emitted when authentication is successful/unsuccessful
    # authentication_done_signal = pyqtSignal(bool)

    def __init__(self):
        super(videoThread, self).__init__()
        self.capture = False
        self.running = True
    
    def update_video_path(self,video_path):
        self.video_path=video_path

    def run(self):

        try:
            print("initalizing video cap...")
            cap = cv2.VideoCapture(self.video_path)

            while cap.isOpened() and self.running:
                # print("reading image")
                success, frame = cap.read()
                if success:
                    # frame=cv2.resize(frame,(1080,720))
                    image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
                    # print("Sending frames cv2 -> pixmap")
                    # self.main_object(image)
                    self.image_signal.emit(image)
                    time.sleep(0.1)

            cap.release()
            self.running=False

        except:
            traceback.print_exc()
            exit()
    def stop_thread(self):
        self.running = False