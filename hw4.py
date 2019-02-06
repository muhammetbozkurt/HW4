import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QGroupBox, QAction, QFileDialog,qApp,QErrorMessage 
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import draw_seg,harris_cop

class App(QMainWindow):

    def __init__(self):
        super(App, self).__init__()
        self.inputLoaded = False
        self.nameOfInput =""
        self.title = 'Main'
        self.setGeometry(50,50,1100,700)
        openInputAction=QAction("&Open input",self)
        openInputAction.triggered.connect(self.openInputImage)
        exitAction = QAction("&Exit",self)
        exitAction.triggered.connect(sys.exit)

        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(openInputAction)
        fileMenu.addAction(exitAction)
        self.initUI()

    def harris_clicked(self):
        error_message = QErrorMessage(self)
        if not self.inputLoaded :
            error_message.setWindowTitle("Input error")
            error_message.showMessage('First load input ')

        else:
            I = cv2.imread(self.nameOfInput)
            R = harris_cop(I)
            cv2.imwrite("res.jpg",R)
            self.labelResult.setPixmap(QPixmap("res.jpg"))

    def segmentation_clicked(self):
        error_message = QErrorMessage(self)
        if not self.inputLoaded :
            error_message.setWindowTitle("Input error")
            error_message.showMessage('First load input ')

        else:
            I = cv2.imread(self.nameOfInput)
            draw_seg(I)
            cv2.imwrite("r.jpg",I)
            self.labelResult.setPixmap(QPixmap("r.jpg"))
            

    def openInputImage(self):
        nameOfInput = QFileDialog.getOpenFileName(self,"Open input")
        self.nameOfInput=nameOfInput[0]
        if self.nameOfInput:
            self.inputLoaded=True
            self.labelINPUT.setPixmap(QPixmap(self.nameOfInput))

    def initUI(self):
        extractAction = QAction('harris corner',self)
        extractAction.triggered.connect(self.harris_clicked)
        extractAction1 = QAction('segmentation',self)
        extractAction1.triggered.connect(self.segmentation_clicked)
        self.toolbar = self.addToolBar("Extraction")
        self.toolbar.addAction(extractAction)
        self.toolbar.addAction(extractAction1)

        self.labelINPUT=QLabel(self)
        self.labelINPUT.setGeometry(50,70,500,500)
        self.labelResult=QLabel(self)
        self.labelResult.setGeometry(650,70,500,500)

        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())