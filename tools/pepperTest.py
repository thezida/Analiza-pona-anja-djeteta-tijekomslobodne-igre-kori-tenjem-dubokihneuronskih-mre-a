#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Shows how to show live images from Nao using PyQt"""

import qi
import argparse
import sys
from PyQt4.QtGui import QWidget, QImage, QApplication, QPainter
from PyQt4 import QtGui, QtCore
import vision_definitions
import _init_paths
from fast_rcnn.config import cfg
import network
import Image
import numpy as np
import cv2
import time
from trackingUtils import detector_utils as detector_utils
import tensorflow as tf
import xml.etree.ElementTree as ET
import lxml.etree as etree
import datetime, time
from utils.timer import Timer

def main(session, robot_ip, port, imageDetector, detection_graph, trackingSess):
    """
    This is a tiny example that shows how to show live images from Nao using PyQt.
    You must have python-qt4 installed on your system.
    """
    CameraID = 0

    # Get the service ALVideoDevice.

    video_service = session.service("ALVideoDevice")
    app = QApplication([robot_ip, port])
    myWidget = ImageWidget(video_service, CameraID, imageDetector, detection_graph, trackingSess)
    window = Window(myWidget)
    window.show()
    sys.exit(app.exec_())


class Window(QtGui.QWidget):
    def __init__(self, imageWidget):
        QtGui.QWidget.__init__(self)
        self.button = QtGui.QPushButton('Detect objects', self)
        self.button.clicked.connect(self.handleButton)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(imageWidget)
        self.resize(320, 270)
        self.imageWidget = imageWidget

    def handleButton(self):
        alImage, alImage3d = self.imageWidget.getImage()

        frame = ET.SubElement(self.imageWidget.body, 'frame')
        timer = Timer()
        timer.tic()
        im = self.imageWidget.imageDetector.detect(self.imageWidget.getImageFromAlValue(alImage), self.imageWidget.get3DImageFromAlValue(alImage3d), self.imageWidget.detection_graph, self.imageWidget.trackingSess,frame)
        timer.toc()
        deltaTime = ET.SubElement(frame, 'delta-time')
        deltaTime.text = str(timer.total_time)
        #self._alImage = self.getAlValueFromImage(im, self._alImage)

class ImageWidget(QWidget):
    """
    Tiny widget to display camera images from Naoqi.
    """
    def __init__(self, video_service, CameraID, imageDetector, detection_graph, trackingSess, parent=None):
        """
        Initialization.
        """
        self.adrfp = ET.Element('adrfp')
        self.adrfp_header(self.adrfp, "experiment_session")
        self.body = ET.SubElement(self.adrfp, 'body')

        QWidget.__init__(self, parent)
        self.video_service = video_service
        self._image = QImage()
        self.setWindowTitle('Robot')

        self.trackingSess = trackingSess
        self.detection_graph = detection_graph

        self._imgWidth = 320
        self._imgHeight = 240
        self._cameraID = CameraID
        self.imageDetector = imageDetector
        self.resize(self._imgWidth, self._imgHeight)

        # Our video module name.
        self._imgClient = ""

        # This will contain this alImage we get from Nao.
        self._alImage = None

        self._registerImageClient()

        # Trigget 'timerEvent' every 100 ms.
        self.startTimer(100)

    def getImage(self):
        return self._alImage, self._alImage3d

    def adrfp_header(self, elT, name):
        header = ET.SubElement(elT, 'header')
        name_xml = ET.SubElement(header, 'name')
        name_xml.text = name
        date = ET.SubElement(header, 'date')
        date.text = datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d - %H:%M:%S')

    def _registerImageClient(self):
        """
        Register our video module to the robot.
        """
        resolution = vision_definitions.kQVGA  # 320 * 240
        colorSpace = vision_definitions.kRGBColorSpace
        self._imgClient = self.video_service.subscribe("_client", resolution, colorSpace, 5)

        resolution = vision_definitions.kQVGA
        colorSpace = vision_definitions.kYuvColorSpace

        self._imgClient2 = self.video_service.subscribe("_client2", resolution, colorSpace, 5)

        # Select camera.
        self.video_service.setParam(vision_definitions.kCameraSelectID, self._cameraID)


    def _unregisterImageClient(self):
        """
        Unregister our naoqi video module.
        """
        if self._imgClient != "":
            self.video_service.unsubscribe(self._imgClient)
            self.video_service.unsubscribe(self._imgClient2)


    def paintEvent(self, event):
        """
        Draw the QImage on screen.
        """
        painter = QPainter(self)
        painter.drawImage(painter.viewport(), self._image)


    def _updateImage(self):
        """
        Retrieve a new image from Nao.
        """
        self._alImage = self.video_service.getImageRemote(self._imgClient)

        self.video_service.setParam(vision_definitions.kCameraSelectID, 2)

        self._alImage3d = self.video_service.getImageRemote(self._imgClient2)

        self.video_service.setParam(vision_definitions.kCameraSelectID, 0)

        self._image = QImage(self._alImage[6],           # Pixel array.
                             self._alImage[0],           # Width.
                             self._alImage[1],           # Height.
                             QImage.Format_RGB888)
        time.sleep(0.1)

    def get3DImageFromAlValue(self, alImage):
        width = alImage[0]
        height = alImage[1]
        pixels = alImage[6]
        im = np.zeros((height, width, 1), dtype=np.uint8)
        index = 0
        for y in range(height):
            for x in range(width):
                triplet = np.zeros(1, dtype=np.uint8)
                #print "prije:"+str(triplet)
                triplet[0] = pixels[index]
                #print "poslje:"+str(triplet)
                im[y][x] = triplet
                #print im[y][x]
                index += 1
        return im

    def getImageFromAlValue(self, alImage):
        width = alImage[0]
        height = alImage[1]
        pixels = alImage[6]
        im = np.zeros((height, width, 3), dtype=np.uint8)
        index = 0
        for y in range(height):
            for x in range(width):
                triplet = np.zeros(3, dtype=np.uint8)
                triplet[0] = pixels[index+2]
                triplet[1] = pixels[index+1]
                triplet[2] = pixels[index]
                im[y][x] = triplet
                index += 3
        return im

    def getAlValueFromImage(self, im, alImage):
        width = alImage[0]
        height = alImage[1]
        index = 0
        for y in range(height):
            for x in range(width):
                triplet = im[y][x]
                alImage[6][index] = triplet[0]  
                alImage[6][index+1] = triplet[1]
                alImage[6][index+2] = triplet[2]
                index += 3
        return alImage

    def timerEvent(self, event):
        """
        Called periodically. Retrieve a nao image, and update the widget.
        """
        self._updateImage()
        self.update()


    def __del__(self):
        """
        When the widget is deleted, we unregister our naoqi video module.
        """
        self._unregisterImageClient()
        output_path = "experiment_session.adrfp"
        tree = ET.ElementTree(self.adrfp)
        tree.write(output_path)
        x = etree.parse(output_path)
        adrfp_string = etree.tostring(x, pretty_print=True)
        with open(output_path, "w") as fw:
            fw.write(adrfp_string)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='FRCNN pepper')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]', default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path', default=' ')
    parser.add_argument("--ip", type=str, default="192.168.2.109", help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559, help="Naoqi port number") 

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    print "Initializing frcnn..."
    imageDetector = network.frcnnImageDetections(args)

    print "Initializing handtracking..."
    detection_graph, trackingSess = detector_utils.load_inference_graph()

    print "Starting naoqi session..."
    NAOQIsess = qi.Session()
    try:
        NAOQIsess.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    main(NAOQIsess, args.ip, args.port, imageDetector, detection_graph, trackingSess)