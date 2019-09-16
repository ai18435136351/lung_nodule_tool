import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from outline_gen import process
from data_read import data_read
from volume import volume_count, weight_count

path_data = r'C:\Users\dell\Desktop\2019.5.6\data_all'
path_mha_image = r'C:\Users\dell\Desktop\2019.5.6\all_npy_and_image\Cv2_image_mha'
path_mhd_image = r'C:\Users\dell\Desktop\2019.5.6\all_npy_and_image\Cv2_image_mhd'
path = r'C:\Users\dell\Desktop\2019.5.6\img_process\process'


class picture(QMainWindow):

    def __init__(self):
        super(picture, self).__init__()
        self.initUI()
        self.num1 = 0
        with open('style.qss', 'r') as q:
            self.setStyleSheet(q.read())

    def initUI(self):
        self.use_palette()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('文件')
        newAct = QAction('文件读取', self)
        newAct.triggered.connect(self.showDialog)
        fileMenu.addAction(newAct)
        menubar1 = self.menuBar()
        fileMenu = menubar1.addMenu('图片预测')
        newAct1 = QAction('图片读取', self)
        # newAct1.triggered.connect(self.showDialog)
        fileMenu.addAction(newAct1)

        self.setFixedSize(1130, 870)
        self.setWindowTitle("肺结节检测程序")

        self.label = QLabel(self)
        self.label.setText("显示图片")  # 设置显示文字
        self.label.setFixedSize(700, 700)
        self.label.move(40, 120)
        # 边框背景色
        # self.label.setStyleSheet("QLabel{background:white;}"
        #                          "QLabel{color:rgb(300,300,300,100);font-size:10px;font-weight:bold;font-family:宋体;}"
        #                          )

        self.label_IOU = QLabel(self)
        self.label_IOU.setText(' IOU:')
        self.label_IOU.setFixedSize(360, 35)
        self.label_IOU.move(780, 200)
        # self.label_IOU.setStyleSheet("QLabel{background:white;}"
        #                           "QLabel{color:rgb(300,300,300,100);font-size:15px;font-weight:bold;font-family:宋体;}"
        #                           )
        self.label_location = QLabel(self)
        self.label_location.setText(' 鼠标点击位置:')
        self.label_location.setFixedSize(360, 35)
        self.label_location.move(780, 270)
        # self.label_location.setStyleSheet("QLabel{background:white;}"
        #                           "QLabel{color:rgb(300,300,300,100);font-size:15px;font-weight:bold;font-family:宋体;}")

        self.label_volume = QLabel(self)
        self.label_volume.setText(' 结节体积大小:')
        self.label_volume.setFixedSize(360, 35)
        self.label_volume.move(780, 340)
        # self.label_volume.setStyleSheet("QLabel{background:white;}"
        #                                   "QLabel{color:rgb(300,300,300,100);font-size:15px;font-weight:bold;font-family:宋体;}")

        self.label_weight = QLabel(self)
        self.label_weight.setText(' 结节重量大小:')
        self.label_weight.setFixedSize(360, 35)
        self.label_weight.move(780, 410)
        # self.label_weight.setStyleSheet("QLabel{background:white;}"
        #                                 "QLabel{color:rgb(300,300,300,100);font-size:15px;font-weight:bold;font-family:宋体;}")

        self.label_vacuole = QLabel(self)
        self.label_vacuole.setText(' 去除空泡后体积大小:')
        self.label_vacuole.setFixedSize(360, 35)
        self.label_vacuole.move(780, 480)
        # self.label_vacuole.setStyleSheet("QLabel{background:white;}"
        #                                 "QLabel{color:rgb(300,300,300,100);font-size:15px;font-weight:bold;font-family:宋体;}")

        self.label_dia = QLabel(self)
        self.label_dia.setText(' 最大直径大小:')
        self.label_dia.setFixedSize(360, 35)
        self.label_dia.move(780, 550)
        # self.label_dia.setStyleSheet("QLabel{background:white;}"
        #                                  "QLabel{color:rgb(300,300,300,100);font-size:15px;font-weight:bold;font-family:宋体;}")
        # 设置按钮
        self.btn = QPushButton(self)
        self.btn.setText("图片显示")
        self.btn.move(50, 60)
        # next按钮
        self.btn_next = QPushButton(self)
        self.btn_next.setText("下一张图片 -->")
        self.btn_next.move(945, 130)
        # last按钮
        self.btn_last = QPushButton(self)
        self.btn_last.setText("<-- 上一张图片")
        self.btn_last.move(795, 130)
        # 鼠标点击图片处理
        self.btn_mouse = QPushButton(self)
        self.btn_mouse.setText("鼠标点击处理")
        self.btn_mouse.move(350, 60)
        # IOU值
        self.btn_IOU = QPushButton(self)
        self.btn_IOU.setText("IOU值显示")
        self.btn_IOU.move(200, 60)
        # 体积计算
        self.btn_volume = QPushButton(self)
        self.btn_volume.setText("结节体积计算")
        self.btn_volume.move(500, 60)
        # 重量计算
        self.btn_weight = QPushButton(self)
        self.btn_weight.setText("结节重量计算")
        self.btn_weight.move(650, 60)

        self.btn_vacuole = QPushButton(self)
        self.btn_vacuole.setText("空泡体积去除计算")
        self.btn_vacuole.move(800, 60)

        self.btn_vacuole = QPushButton(self)
        self.btn_vacuole.setText("最大直径计算")
        self.btn_vacuole.move(950, 60)


    def showDialog(self):

        self.num1 += 1
        print('num1', self.num1)
        self.name_path, fname = QFileDialog.getOpenFileName(self, 'Open file', path_data)
        print(str(self.name_path))
        self.name1, self.name, self.image_num = data_read(str(self.name_path))
        print(self.name_path[len(self.name_path) - 4:len(self.name_path)])
        if self.name_path[len(self.name_path) - 4:len(self.name_path)] == '.mha':
            # print(self.name_path[len(self.name_path) - 4:len(self.name_path)])
            self.btn.clicked.connect(self.openimage_mha)
        else:
            self.btn.clicked.connect(self.openimage_mhd)

    def openimage_mha(self):
        # mha首位图片加载

        self.num = 0
        self.type = 'mha'
        self.img_mha = path_mha_image + '.' + '/' + self.name1 + '.' + '/' + self.name + '.' + '/' + self.name1 + '0.png'
        img3 = QtGui.QPixmap(self.img_mha).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img3)

        self.btn_next.clicked.connect(self.next_image)
        self.btn_last.clicked.connect(self.last_image)

    def openimage_mhd(self):
        # mhd首位图片加载

        self.num = 0
        print('self.num', self.num)
        self.type = 'mhd'
        self.img_mhd = path_mhd_image + '.' + '/' + self.name1 + '.' + '/' + self.name + '.' + '/' + self.name1 + '0.png'
        img3 = QtGui.QPixmap(self.img_mhd).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img3)

        self.btn_next.clicked.connect(self.next_image)
        self.btn_last.clicked.connect(self.last_image)

    def next_image(self):   # 下一张图片

        self.num += 1
        num2 = 1
        for i in range(2, self.num1 + 1):
            num2 += i
        print(self.num/num2)
        num3 = int(self.num/num2)

        if self.num <= self.image_num:
            if self.type == 'mha':
                path_type = path_mha_image
            else:
                path_type = path_mhd_image
            img_next = path_type + '.' + '/' + self.name1 + '.' + '/' + self.name + '.' + '/' + self.name1 + str(num3) + '.png'
            img_next = QtGui.QPixmap(img_next).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(img_next)

    def last_image(self):   # 上一张图片

        self.num -= 1
        num2 = 1
        for i in range(2, self.num1 + 1):
            num2 += i
        num3 = int(self.num/num2)
        print(num3)
        if self.type == 'mha':
            path_type = path_mha_image
        else:
            path_type = path_mhd_image
        if self.num >= 0:
            img_next = path_type + '.' + '/' + self.name1 + '.' + '/' + self.name + '.' + '/' + self.name1 + str(num3) + '.png'
            img_next = QtGui.QPixmap(img_next).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(img_next)

    def mousePressEvent(self, e):   # 鼠标点击位置
        self.x = e.pos().x()
        self.y = e.pos().y()
        i_num = int((self.x - 30) * 1024/700)
        j_num = int((self.y - 100) * 1024/700)
        self.label_location.setText('鼠标点击位置:' + str(j_num) + ',' + str(i_num))
        self.btn_mouse.clicked.connect(self.mouse_click)

    def mouse_click(self):          # 鼠标点击处理

        if self.type == 'mhd':
            print(self.name1, self.name,self.name1 + str(self.num))
            self.locationx = (self.x - 30) * 1024/700
            self.locationy = (self.y - 100) *1024/700

            print(self.locationx,self.locationy)

            self.acc = process(str(self.name1), str(self.name), str(self.name1 + str(self.num)), int(self.locationx), int(self.locationy))
            self.IOUimage = str(path + '.' + '/' + str(self.name1 + str(self.num)) + '.png')

            jpg = QtGui.QPixmap(self.IOUimage).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)
            self.btn_IOU.clicked.connect(self.IOU)
            self.btn_volume.clicked.connect(self.Volume)    # 体积计算
            self.btn_weight.clicked.connect(self.Weight)    # 重量计算

    def IOU(self):                  # IOU值处理
        self.label_IOU.setText('IOU:' + str(self.acc))

    def Volume(self):
        self.volume = volume_count(str(self.name1), str(self.name), int(self.locationx), int(self.locationy))
        volume_all = self.volume * (1024/798) * (1024/798) * 0.03089
        self.label_volume.setText('体积大小:' + str(int(volume_all)) + 'mm**3')

    def Weight(self): #
        self.weight = weight_count(str(self.name1), str(self.name), int(self.locationx), int(self.locationy))
        weight_all = self.weight * (1024/798) * (1024/798) * 1000 / 1000000 * 0.03089
        self.label_weight.setText('重量大小:' + str(int(weight_all)) + 'mg')

    def use_palette(self):
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("./image/main_skin_04.jpg")))
        self.setPalette(window_pale)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
