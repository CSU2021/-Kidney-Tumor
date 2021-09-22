import os
import sys
import  cv2
import time
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageQt
from Ui_mainwindow_1 import Ui_Form
#from predictnii import predict, kindey2d3dmerge, Proresult, changeWL
from predictnii import changeWL
import vtk_first_try
#import vtk
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np


n=0
max_n=425
img=cv2.imread(os.path.split(os.path.realpath(__file__))[0]+r'/imgs/blank.png')
#img=cv2.imread('C:/Users/szh/Desktop/QT/imgs/blank.png')
imglst_1=[]
imglst_2=[]
imglst_3=[]
imglst_4=[]
leftMaxlenList=[]
leftAreaList=[]
rightMaxlenList=[]
rightAreaList=[]

class L(QLabel):
    def __init__(self, parent):
        global path
        super().__init__(parent=parent)
        self.setStyleSheet('QFrame {background-color:black;}')
        self.setGeometry(QtCore.QRect(290, 150, 780, 780))
        self.img1 = ImageQt.toqpixmap(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.scaled_img = self.img1.scaled(self.size())
        self.point = QPoint(0, 0)
        self.lastPos= QPoint(self.width()/2,self.height()/2)#十字线复原
    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        self.draw_img(painter)
        self.line(painter)
        painter.end()
        
    def line(self,painter):
        pen=QPen()
        pen.setWidth(2)
        pen.setStyle(Qt.DashDotLine)
        pen.setColor(Qt.blue)
        painter.setPen(pen)
        painter.drawLine(0, self.lastPos.y(), self.width(), self.lastPos.y())
        painter.drawLine(self.lastPos.x(), 0, self.lastPos.x(), self.height())
        
    def draw_img(self, painter):
        painter.drawPixmap(self.point, self.scaled_img)

    def mouseMoveEvent(self, e):  # 重写移动事件
        if self.left_click:
            self._endPos = e.pos() - self._startPos
            self.point = self.point + self._endPos
            self._startPos = e.pos()
            self.lastPos=e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self._startPos = e.pos()
            self.lastPos=e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.lastPos=e.pos()
            #self.left_click = False
            self.repaint()
        elif e.button() == Qt.RightButton:
            self.point = QPoint(0, 0)
            self.scaled_img = self.img1.scaled(self.size())
            self.lastPos= QPoint(self.width()/2,self.height()/2)
            self.repaint()

    def wheelEvent(self, e):
        if e.angleDelta().y() > 0:
            # 放大图片
            self.scaled_img = self.img1.scaled(self.scaled_img.width()-20, self.scaled_img.height()-20)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() + 20)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() + 20)
            self.point = QPoint(new_w, new_h)
            self.repaint()
        elif e.angleDelta().y() < 0:
            # 缩小图片
            self.scaled_img = self.img1.scaled(self.scaled_img.width()+20, self.scaled_img.height()+20)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() - 20)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() - 20)
            self.point = QPoint(new_w, new_h)
            self.repaint()

    '''def resizeEvent(self, e):
        if self.parent is not None:
            self.scaled_img = self.img1.scaled(self.size())
            self.point = QPoint(0, 0)
            self.update()'''

class Mainwindow(QWidget):
    def __init__(self):
        super().__init__()
        self.label=L(self)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.phone)
        self.ui.pushButton_2.clicked.connect(self.download)
        self.ui.pushButton_3.clicked.connect(self.deal)
        self.ui.pushButton_9.clicked.connect(self.help)
        self.ui.pushButton_10.clicked.connect(self.left)
        self.ui.pushButton_11.clicked.connect(self.right)
        self.ui.pushButton_12.clicked.connect(self.change)
        self.ui.lineEdit_3.returnPressed.connect(self.enter)
        self.ui.horizontalSlider.valueChanged.connect(self.slider)
        self.ui.checkBox.stateChanged.connect(self.check)
        self.ui.checkBox_2.stateChanged.connect(self.check)
        self.vtkWindow = QVTKRenderWindowInteractor(self)
        self.vtkWindow.setGeometry(1090, 150, 780, 780)
        self.ui.label_13.raise_()

    def phone(self):
        QMessageBox.about(self,'联系电话','联系电话')
    def help(self):
        QMessageBox.about(self,'帮助','帮助')
    def change(self):
        global path
        global resultk
        global resultt
        global imglst_1
        global imglst_2
        global imglst_3
        global imglst_4
        window=self.ui.lineEdit_2.text()
        level=self.ui.lineEdit.text()
        self.ui.lineEdit_4.setText('窗位窗宽转换中······\n 请稍后')
        imglst_1, imglst_2, imglst_3, imglst_4=changeWL(int(window),int(level),path,resultk, resultt)
        self.check()
        self.ui.lineEdit_4.setText('窗位窗宽转换完成！')
    def download(self):
        print('download')
        name = (QFileDialog.getSaveFileName(None, "Save File",
                            "szh.pdf", "*.pdf"))[0]
        pdfFile = QFile(name)
        #打开要写入的pdf文件
        pdfFile.open(QIODevice.WriteOnly)

        #创建pdf写入器
        pPdfWriter = QPdfWriter(pdfFile)
        #设置纸张为A4
        pPdfWriter.setPageSize(QPagedPaintDevice.A4)
        #设置纸张的分辨率为300,因此其像素为3508X2479
        pPdfWriter.setResolution(300)
        pPdfWriter.setPageMargins(QMarginsF(60, 60, 60, 60))

        pPdfPainter = QPainter(pPdfWriter)

        # 标题上边留白
        iTop = 100

        #文本宽度2100
        iContentWidth = 2100

        # 标题,22号字
        font = QFont()
        font.setFamily("simhei.ttf")
        fontSize = 22
        font.setPointSize(fontSize)

        pPdfPainter.setFont(font)
        pPdfPainter.drawText(QRect(0, iTop, iContentWidth, 90), Qt.AlignHCenter, "我是标题我骄傲")

        # 内容,16号字，左对齐
        fontSize = 16
        font.setPointSize(fontSize)
        pPdfPainter.setFont(font)

        iTop += 90
        pPdfPainter.drawText(QRect(0, iTop, iContentWidth, 60), Qt.AlignLeft, "1、目录一")
        iTop += 90
        # 左侧缩进2字符
        iLeft = 120
        pPdfPainter.drawText(QRect(iLeft, iTop, iContentWidth - iLeft, 60), Qt.AlignLeft, "我的目录一的内容。")
        iTop += 90
        pPdfPainter.drawText(QRect(0, iTop, iContentWidth, 60), Qt.AlignLeft, "2、目录二")
        iTop += 90
        pPdfPainter.drawText(QRect(iLeft, iTop, iContentWidth - iLeft, 60), Qt.AlignLeft, "我的目录一的内容")

        pPdfPainter.end()
        pdfFile.close()
    def left(self):
        global n
        n=(n-1)%max_n
        self.ui.lineEdit_3.setText(str(n+1))
        self.ui.horizontalSlider.setValue(n+1)
        
        self.check()
    def right(self):
        global n
        n=(n+1)%max_n
        self.ui.lineEdit_3.setText(str(n+1))
        self.ui.horizontalSlider.setValue(n+1)
        
        self.check()
    def enter(self):
        global n
        n=int(self.ui.lineEdit_3.text())-1
        self.ui.horizontalSlider.setValue(n+1)
        
        self.check()
    def slider(self):
        global n
        n=int(self.ui.horizontalSlider.value())-1
        self.ui.lineEdit_3.setText(str(n+1))
        
        self.check()
    def deal(self):
        global n
        global img
        global imglst_1
        global imglst_2
        global imglst_3
        global imglst_4
        global leftMaxlenList
        global leftAreaList
        global rightMaxlenList
        global rightAreaList
        global path
        global resultk
        global resultt
        
        n=0
        self.ui.lineEdit_3.setText(str(n+1))
        self.ui.horizontalSlider.setValue(n+1)
        self.ui.checkBox.setChecked(False)
        self.ui.checkBox_2.setChecked(False)
        self.ui.lineEdit_4.setText('处理中······（预计等待3分钟）')
        path= (QtWidgets.QFileDialog.getOpenFileName(None, '浏览', os.path.split(os.path.realpath(__file__))[0]))[0]

        self.vtkWindow.Initialize()
        self.vtkWindow.Start()
        ren = vtk.vtkRenderer()
        self.vtkWindow.GetRenderWindow().AddRenderer(ren)
        coneActor, coneMapper = vtk_first_try.open_nii(os.path.split(os.path.realpath(__file__))[0]+'/imaging.nii.gz')
        ren.AddVolume(coneActor)
        
        #调用函数
        resultk = np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/kidney.npy')
        resultt = np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/tumor.npy')
        imglst_1=np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/k0t0.npy')
        imglst_2=np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/k1t0.npy')
        imglst_3=np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/k0t1.npy')
        imglst_4=np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/k1t1.npy')
        leftMaxlenList=np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/leftmax.npy')
        leftAreaList=np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/leftarea.npy',allow_pickle=True)
        rightMaxlenList=np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/rightmax.npy')
        rightAreaList=np.load(os.path.split(os.path.realpath(__file__))[0]+'/output/rightarea.npy',allow_pickle=True)
        
        # resultk = np.load('C:/Users/szh/Desktop/QT/output/kidney.npy')
        # resultt = np.load('C:/Users/szh/Desktop/QT/output/tumor.npy')
        # imglst_1=np.load('C:/Users/szh/Desktop/QT/output/k0t0.npy')
        # imglst_2=np.load('C:/Users/szh/Desktop/QT/output/k1t0.npy')
        # imglst_3=np.load('C:/Users/szh/Desktop/QT/output/k0t1.npy')
        # imglst_4=np.load('C:/Users/szh/Desktop/QT/output/k1t1.npy')
        # leftMaxlenList=np.load('C:/Users/szh/Desktop/QT/output/leftmax.npy')
        # leftAreaList=np.load('C:/Users/szh/Desktop/QT/output/leftarea.npy',allow_pickle=True)
        # rightMaxlenList=np.load('C:/Users/szh/Desktop/QT/output/rightmax.npy')
        
        self.ui.lineEdit_4.setText('处理完成！')
        self.check()
        self.ui.tableWidget_2.item(0,0).setText('Zhangsan')
        self.ui.tableWidget_2.item(1,0).setText('2020/05/03')
        self.ui.tableWidget_2.item(2,0).setText('[85,512,512]')
        self.ui.tableWidget_2.item(3,0).setText('[5,0.664062,0.664062]')
        self.ui.tableWidget_2.item(4,0).setText('[0,0,0]')
        self.ui.tableWidget_2.item(5,0).setText('22282240')
    def check(self):
        global n
        global img
        global imglst_1
        global imglst_2
        global imglst_3
        global imglst_4
        global leftMaxlenList
        global leftAreaList
        global rightMaxlenList
        global rightAreaList
        self.ui.tableWidget.item(0,0).setText(str(int(leftMaxlenList[n])))
        self.ui.tableWidget.item(0,1).setText(str(rightMaxlenList[n]))
        self.ui.tableWidget.item(1,0).setText(str(leftAreaList[n][0]))
        #self.ui.tableWidget.item(1,1).setText(str(rightAreaList[n][0]))
        self.ui.tableWidget.item(1,1).setText(str(0))
        if not self.ui.checkBox.isChecked() and  not self.ui.checkBox_2.isChecked():
            img=imglst_1[n]
            self.up()
        elif self.ui.checkBox.isChecked() and not self.ui.checkBox_2.isChecked():
            img=imglst_2[n]
            self.up()
        elif not self.ui.checkBox.isChecked() and self.ui.checkBox_2.isChecked():
            img=imglst_3[n]
            self.up()
        elif self.ui.checkBox.isChecked() and self.ui.checkBox_2.isChecked():
            img=imglst_4[n]
            self.up()
    def up(self):
        global img
        self.label.point = QPoint(0, 0)
        self.label.img1=ImageQt.toqpixmap(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.label.scaled_img = self.label.img1.scaled(self.label.size())
        self.label.lastPos= QPoint(self.label.width()/2,self.label.height()/2)
        self.label.repaint()
if __name__ == '__main__':
    app = QApplication([])
    app.setWindowIcon(QIcon(os.path.split(os.path.realpath(__file__))[0]+r'/imgs/csu.png'))
    #app.setWindowIcon(QIcon('C:/Users/szh/Desktop/QT/imgs/csu.png'))
    mainwindow=Mainwindow()
    mainwindow.show()
    app.exec_()