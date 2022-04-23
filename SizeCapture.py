import sys

import os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import uic
from numpy import ndarray

import cv2
import image2mesh

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))



#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("sizecapture_main.ui")[0]
#화면을 띄우는데 사용되는 Class 선언

class ResultWindow(QDialog):
    def __init__(self, parent, size_dict,heightInput):
        super(ResultWindow,self).__init__(parent)
        self.setWindowTitle("Result")
        uic.loadUi("sizecapture_result.ui", self)
        self.heightInput = heightInput
        self.Result_table.addItems(self.getSizeItem(size_dict))
        self.show()

        self.Result_close.clicked.connect(self.clickClose)

    def clickClose(self):
        self.close()

    def getSizeItem(self,size_dict):
        height = int(self.heightInput)
        shoulder = size_dict['height']/size_dict['shoulder']
        shoulder = '어깨길이: '+str(round(height/shoulder*1.2,2))+'cm'

        hip = size_dict['height']/size_dict['hip']
        hip = '골반길이: '+str(round(height/hip*1.2,2))+'cm'

        leg = size_dict['height']/size_dict['leg']
        leg = '다리길이: '+str(round(height/leg,2))+'cm'

        reach = size_dict['height']/size_dict['reach']
        reach = '%-4s'%'팔길이'+' : '+str(round(height/reach,2))+'cm'

        return shoulder,hip,leg,reach


class MainWindow(QMainWindow, form_class):
    loadSignal = pyqtSignal()
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle("sizeCapture")
        self.setupUi(self)
        self.show()
        self.isToggle = False
        self.visImage = 0
        self.meshImage = 0

        # 레이아웃과 타이틀바 위젯 생성
        window_vbox = QVBoxLayout(self.centralWidget())
        window_vbox.setContentsMargins(0,0,0,0)
        title_hbox = QHBoxLayout()
        content_hbox = QHBoxLayout()

        title_hbox.addWidget(MainTitleBar(self))

        content_hbox.addWidget(self.main_frame)
        content_hbox.setContentsMargins(60,50,60,50)
        window_vbox.addLayout(title_hbox)
        window_vbox.addLayout(content_hbox)

        self.Measure.clicked.connect(self.convImage)

        self.loadSignal.connect(self.loading)

        self.filePath = ""
        self.file_search.clicked.connect(self.showFileDialog)
        self.toggle.clicked.connect(self.toggleBtn)


    @pyqtSlot()
    def loading(self):
        # 로딩중일때 다시 클릭하는 경우
        try:
            self.loading
            self.loading.deleteLater()

        # 처음 클릭하는 경우
        except:
            self.loading = loading(self)

    def showFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'demoImage')
        path,type = fname

        if self.isPathFormatImage(path):
            self.filePath = path
            image = cv2.imread(path)
            image = self.getImageResize(image)
            qImg = self.getImageFromCV2(image)
            result = QtGui.QPixmap(qImg)

            self.Image.setPixmap(result)

            self.meshImage = None
            self.visImage = None
        else:
            print('이미지 파일이 아닙니다.')


    def getImageResize(self,cv2image):
        width = cv2image.shape[1]
        height = cv2image.shape[0]
        max_width = 500
        max_height = 400
        ratio = width/height
        if ratio < 1:
            result = cv2.resize(cv2image,dsize=tuple(map(int,(max_height*ratio,max_height))))
        else:
            ratio = height/width
            result = cv2.resize(cv2image,dsize=tuple(map(int,(max_width,max_width*ratio))))
        return result

    def getImageFromCV2(self,cv2image,imgFormat=QImage.Format_RGB888):
        image = cv2image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        qImg = QtGui.QImage(image.data, w, h, w * c,imgFormat)
        return qImg

    def convImage(self):
        if self.filePath != "":
            self.Measure.setEnabled(False)
            self.file_search.setEnabled(False)
            self.toggle.setEnabled(False)
            self.convThread = Thread(self)
            self.convThread.__init__(self)
            self.convThread.filePath = self.filePath
            print(self.filePath)
            self.convThread.threadEvent.connect(self.threadEventHandler)
            self.loadSignal.emit()
            self.convThread.start()

    def toggleBtn(self):
        if self.isToggle:
            anim = QPropertyAnimation(self.toggle, b"geometry")
            anim.setDuration(700)
            anim.setStartValue(QRect(2, 2, 26, 26))
            anim.setEndValue(QRect(23, 2, 26, 26))
            anim.start()
        else:
            anim = QPropertyAnimation(self.toggle,b"geometry")
            anim.setDuration(700)
            anim.setStartValue(QRect(23,2,26,26))
            anim.setEndValue(QRect(2,2,26,26))
            anim.start()
        self.changeImage()
        self.isToggle = not self.isToggle

    def changeImage(self):
        if self.meshImage and self.visImage:
            if self.isToggle:
                self.Image.setPixmap(self.meshImage)
            else:
                self.Image.setPixmap(self.visImage)

    @pyqtSlot(ndarray,ndarray,dict)
    def threadEventHandler(self,vis,mesh,size_dict):
        result_mesh, result_vis = map(self.getImageResize, (mesh, vis))
        result_mesh,result_vis = map(self.getImageFromCV2,(result_mesh,result_vis))
        result_mesh,result_vis = map(QPixmap,(result_mesh,result_vis))

        self.meshImage = result_mesh
        self.visImage = result_vis
        self.changeImage()
        self.Measure.setEnabled(True)
        self.file_search.setEnabled(True)
        self.toggle.setEnabled(True)
        self.loadSignal.emit()
        ResultWindow(self,size_dict,self.heightInput.text())



    def isPathFormatImage(self, path):
        imageFormatList = [
            'bmp', 'gif', 'jpg', 'jpeg', 'png'
        ]
        pathFormat = path.split('.')[-1]
        if pathFormat in imageFormatList:
            return True
        else:
            return False


class Thread(QThread):
    threadEvent = pyqtSignal(ndarray,ndarray,dict)
    def __init__(self,parent=None):
        QThread.__init__(self)
        self.parent = parent
        self.filePath = ""

    def run(self):
        print(self.filePath)
        result_mesh,result_vis,size_dict = image2mesh.getImageMesh(self.filePath)
        print(type(result_vis),type(result_mesh))
        self.threadEvent.emit(result_vis, result_mesh,size_dict)

class MainTitleBar(QWidget):
    """제목 표시줄 위젯"""
    qss = """
        QWidget {
            color: #FFFFFF;
            background: #00ADB5;
            height: 32px;
        }
        QLabel {
            color: #393E46;
            background: #00ADB5;
            font-size: 16px;
            padding: 5px 5px;
        }
        QToolButton:hover{
            background: #FFFFFF;
            color: #00ADB5;
        }
        QToolButton {
            background: #00ADB5;
            border: none;
        }


    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bar_height = 36
        self.parent = parent
        self.has_clicked = False
        self.is_maximized = False
        self.setStyleSheet(self.qss)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)



        label = QLabel("SIZE CAPTURE")
        label.setFixedHeight(self.bar_height)
        icon = self.create_tool_btn('icon.png')
        btn_minimize = self.create_tool_btn('minimize.png')
        btn_minimize.clicked.connect(self.show_minimized)
        btn_close = self.create_tool_btn('close.png')
        btn_close.clicked.connect(self.close)
        layout.addWidget(icon)
        layout.addWidget(label)
        layout.addWidget(btn_minimize)
        layout.addWidget(btn_close)

    def create_tool_btn(self, icon_path):
        """제목표시줄 아이콘 생성"""
        icon = os.path.join(ROOT_PATH, icon_path)
        btn = QToolButton(self)
        btn.setIcon(QIcon(icon))
        btn.setIconSize(QSize(self.bar_height, self.bar_height))
        btn.setFixedSize(self.bar_height, self.bar_height)
        btn.setEnabled(True)
        return btn

    def show_minimized(self):
        """버튼 명령: 최소화"""
        self.parent.showMinimized()

    def close(self):
        """버튼 명령: 닫기"""
        self.parent.close()

    def mousePressEvent(self, event):
        """오버로딩: 마우스 클릭 이벤트
        - 제목 표시줄 클릭시 이동 가능 플래그
        """
        if event.button() == Qt.LeftButton:
            self.parent.is_moving = True
            self.parent.offset = event.pos()

    def mouseMoveEvent(self, event):
        """오버로딩: 마우스 이동 이벤트
        - 제목 표시줄 드래그시 창 이동
        """
        if self.parent.is_moving:
            self.parent.move(event.globalPos() - self.parent.offset)

    def mouseDoubleClickEvent(self, event):
        """오버로딩: 더블클릭 이벤트
        - 제목 표시줄 더블클릭시 최대화
        """
        if self.is_maximized:
            self.parent.showNormal()
            self.is_maximized = False
        else:
            self.parent.showMaximized()
            self.is_maximized = True

FROM_CLASS_Loading = uic.loadUiType("loading.ui")[0]
class loading(QWidget, FROM_CLASS_Loading):

    def __init__(self, parent):
        super(loading, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.show()

        # 동적 이미지 추가
        self.movie = QMovie('loading.gif', QByteArray(), self)
        self.movie.setCacheMode(QMovie.CacheAll)
        # QLabel에 동적 이미지 삽입
        self.label.setMovie(self.movie)
        self.movie.start()

        # 윈도우 해더 숨기기
        self.setWindowFlags(Qt.FramelessWindowHint)

        # 위젯 정중앙 위치

    def center(self):
        size = self.size()
        ph = self.parent().geometry().height()
        pw = self.parent().geometry().width()
        self.move(int(pw / 2 - size.width() / 2), int(ph / 2 - size.height() / 2))





if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)
    #WindowClass의 인스턴스 생성
    myWindow = MainWindow()
    #프로그램 화면을 보여주는 코드
    myWindow.show()
    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()