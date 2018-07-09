#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Shows how to show live images from Nao using PyQt"""

import qi
import argparse
import sys
from PyQt4.QtGui import QWidget, QImage, QApplication, QPainter
from PyQt4 import QtGui, QtCore
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import vision_definitions
import numpy as np
from random import randint
import random


def main(session, robot_ip, port):
    """
    This is a tiny example that shows how to show live images from Nao using PyQt.
    You must have python-qt4 installed on your system.
    """
    CameraID = 0

    # Get the service ALVideoDevice.


    video_service = session.service("ALVideoDevice")
    app = QApplication([robot_ip, port])
    myWidget = ImageWidget(video_service, CameraID)
    window = Window(myWidget)
    window.show()
    sys.exit(app.exec_())
    #myWidget.show()
    #sys.exit(app.exec_())

class Window(QtGui.QWidget):
    def __init__(self, imageWidget):
        self.widget = imageWidget
        self.widget.setParent(self)
        QtGui.QWidget.__init__(self)
        self.button = QtGui.QPushButton('Calculate Distance', self)
        self.button.clicked.connect(self.handleButton)
        imageWidget.mousePressEvent = self.getPos
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(imageWidget)
        self.resize(400, 300)
        self.bbox = []
        self.bbox.append((0,0))
        self.bbox.append((0,0))
        self.index = 0
        self.tryInd = 0

    def handleButton(self):
        im = self.getImageFromAlValue(self.image)
        im3d = self.get3DImageFromAlValue(self.image3d)

        #print im
        image = Image.fromarray(im.astype('uint8'), 'RGB')
        print "Dubina: "+str(self.getDepht(im3d,self.bbox))
        x1 = int(self.bbox[0][0])
        y1 = int(self.bbox[0][1])
        x2 = int(self.bbox[1][0])
        y2 = int(self.bbox[1][1])
        oim = image.crop((x1,y1,x2,y2))
        oim.save("results/basic/"+str(self.tryInd)+".jpg")
        self.tryInd += 1

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
                triplet[2] = pixels[index+2]
                triplet[1] = pixels[index+1]
                triplet[0] = pixels[index]
                im[y][x] = triplet
                index += 3
        return im

    def getDepht(self, im3d, bbox):
        print bbox
        x1 = int(bbox[0][0])
        y1 = int(bbox[0][1])
        x2 = int(bbox[1][0])
        y2 = int(bbox[1][1])
        average = 0.0
        strIm = ""
        num = 0
        filteredImage = {}
        for y in range(y1,y2+1):
            strIm += "|"
            for x in range(x1,x2+1):
                d = im3d[y][x][0]
                strIm += "{:>3}|".format(d)
                if d<=7: continue
                if x%2==1:
                    continue
                filteredImage[(y,x)] = d
                average += 1.0*d
                num += 1
            strIm += "\n"
        print strIm
        c1,c2 = kmeans(filteredImage, 2, 1000, 320, 240)
        if abs(c1-c2)<6:
            depth = average/num
        else:
            depth = min([c1, c2])
        print (average/num + 18.7579)/0.5181 
        print (min([c1, c2]) + 18.7579)/0.5181
        depth = (depth + 18.7579)/0.5181
        
        return depth

    def getPos(self , event):
        x = event.pos().x()
        y = event.pos().y()
        print "{}.th point: ({},{})".format(self.index,x,y)
        self.bbox[self.index] = (x,y)
        self.index += 1
        self.index %= 2

class ImageWidget(QWidget):
    """
    Tiny widget to display camera images from Naoqi.
    """
    def __init__(self, video_service, CameraID, parent=None):
        """
        Initialization.
        """
        QWidget.__init__(self, parent)
        self.video_service = video_service
        self._image = QImage()
        self.setWindowTitle('Robot')

        self._imgWidth = 320
        self._imgHeight = 240
        self._cameraID = CameraID
        self.resize(self._imgWidth, self._imgHeight)
        self.setFixedSize(self._imgWidth, self._imgHeight)
        # Our video module name.
        self._imgClient = ""

        # This will contain this alImage we get from Nao.
        self._alImage = None

        self._registerImageClient()

        # Trigget 'timerEvent' every 100 ms.
        self.startTimer(100)

    def handleButton(self):
        print ('Hello World')

    def _registerImageClient(self):
        """
        Register our video module to the robot.
        """
        resolution = vision_definitions.kQVGA  # 320 * 240
        colorSpace = vision_definitions.kYuvColorSpace
        self._imgClient = self.video_service.subscribe("_client", resolution, colorSpace, 5)

        resolution = vision_definitions.kQVGA
        colorSpace = vision_definitions.kRGBColorSpace
        self._imgClient2 = self.video_service.subscribe("_client2", resolution, colorSpace, 5)
        

        # Select camera.
        self.video_service.setParam(vision_definitions.kCameraSelectID, self._cameraID)

    def setParent(self, parent):
        self.parent = parent

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
        self._alImage = self.video_service.getImageRemote(self._imgClient2)
        self.parent.image = self._alImage
        self.video_service.setParam(vision_definitions.kCameraSelectID, 2)

        self._alImage3d = self.video_service.getImageRemote(self._imgClient)
        #print self._alImage3d[0], self._alImage3d[1]
        self.parent.image3d = self._alImage3d

        self.video_service.setParam(vision_definitions.kCameraSelectID, 0)

        self._image = QImage(self._alImage[6],           # Pixel array.
                             self._alImage[0],           # Width.
                             self._alImage[1],           # Height.
                             QImage.Format_RGB888)


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

def kmeans(values, numOfCentroids, numOfIter, heigth, width):
    labeledValues = {p: 0 for p in values}
    if numOfCentroids<=1:
        return labeledValues

    centroids = []
    for i in range(numOfCentroids):
        centroids.append(values[random.choice(values.keys())])

    print "random centroids: "+str(centroids)
    iteration = 0
    changed = True
    while iteration<=numOfIter and changed==True:
        print "iteration: "+str(iteration)
        changed = False
        oldCentroids = centroids[:]
        for point in values:
            minC = 0
            minDist = abs(values[point] - centroids[0])

            for i in range(1,numOfCentroids):
                dist = abs(values[point] - centroids[i])
                if dist < minDist:
                    minDist = dist
                    minC = i

            labeledValues[point] = centroids[minC]


        #print labeledValues
        #print centroids
        for i in range(numOfCentroids):
            c = centroids[i]
            sumOfValues = 0.0
            numOfValues = 0.0
            for point in values:
                if labeledValues[point] == c:
                    sumOfValues += values[point]*1.0
                    numOfValues += 1.0

            if numOfValues!=0:
                centroids[i] = sumOfValues/numOfValues
            print abs(centroids[i] - oldCentroids[i])
            if abs(centroids[i] - oldCentroids[i])>0.001:
                changed = True
        #print centroids
        iteration += 1
        print changed
    print centroids
    #return labeledValues
    return centroids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session, args.ip, args.port)