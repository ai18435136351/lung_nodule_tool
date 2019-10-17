import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from outline_gen import process
from data_read import data_read
from volume import volume_count, weight_count
from maker import generate
import os
import tkinter

path_data = r'mha_mhd'
path_mha_image = r'data_mha_mhd/mha_png'
path_mhd_image = r'data_mha_mhd/mhd_png'
path = r'img_process/process'


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
        newAct = QAction('mha/mhd文件读取', self)
        newAct.triggered.connect(self.showDialog)
        newAct0 = QAction('已切片文件读取', self)
        newAct0.triggered.connect(self.showDialog1)
        newAct2 = QAction('已生成文件读取', self)
        newAct2.triggered.connect(self.showDialog2)
        fileMenu.addAction(newAct)
        fileMenu.addAction(newAct0)
        fileMenu.addAction(newAct2)
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

        self.btn_batch = QPushButton(self)
        self.btn_batch.setText("图片批处理")
        self.btn_batch.move(950, 60)

        self.btn_generate = QPushButton(self)
        self.btn_generate.setText("生成病历报告")
        self.btn_generate.clicked.connect(lambda: generate())
        self.btn_generate.move(780, 650)

    # 显示打开mhd,mha文件界面
    def showDialog(self):

        self.num1 += 1
        print('num1', self.num1)
        self.name_path, fname = QFileDialog.getOpenFileName(self, 'Open file', path_data)
        print('self.name_path =', str(self.name_path))
        self.name1, self.name, self.image_num = data_read(str(self.name_path))
        print(self.name_path[len(self.name_path) - 4:len(self.name_path)])
        if self.name_path[len(self.name_path) - 4:len(self.name_path)] == '.mha':
            # print(self.name_path[len(self.name_path) - 4:len(self.name_path)])
            self.btn.clicked.connect(self.openimage_mha)
            reply = QMessageBox.information(self, '提示', '读取成功，请点击“显示图片”以显示', QMessageBox.Yes)
        elif self.name_path[len(self.name_path) - 4:len(self.name_path)] == '.mhd':
            self.btn.clicked.connect(self.openimage_mhd)
            reply = QMessageBox.information(self, '提示', '读取成功，请点击“显示图片”以显示', QMessageBox.Yes)
        else:
            reply = QMessageBox.information(self, '提示', '读取失败，请重试', QMessageBox.Yes)

    # 显示打开文件界面
    def showDialog1(self):
        self.file_path, fname1 = QFileDialog.getOpenFileNames(self, path_mha_image)
        self.image_num = len(self.file_path)
        file_dir = self.file_path[0].split('/')
        self.name1 = file_dir[5]
        self.name = file_dir[6]
        reply = QMessageBox.information(self, '提示', '读取成功，请点击“显示图片”以显示', QMessageBox.Yes)
        self.btn.clicked.connect(self.openimage)

    def showDialog2(self):
        self.file_path, fname1 = QFileDialog.getOpenFileNames(self, path_mha_image)
        self.image_num = len(self.file_path)
        file_dir = self.file_path[0].split('/')
        self.name1 = file_dir[5]
        self.name = file_dir[6]
        self.btn.clicked.connect(self.openimage_finished)
        reply = QMessageBox.information(self, '提示', '读取成功，请点击“显示图片”以显示', QMessageBox.Yes)

    # 打开已生成的图片
    def openimage_finished(self):
        self.num = 0
        self.total_num = len(self.file_path)
        self.image_index = 0
        self.type = 'mhd'
        self.IOUimage = self.file_path[self.image_index]
        jpg = QtGui.QPixmap(self.IOUimage).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.btn_next.clicked.connect(self.next_IOUI)
        self.btn_last.clicked.connect(self.last_IOUI)

    # 打开图片
    def openimage(self):
        self.num = 0
        self.total_num = len(self.file_path)
        self.image_index = 0
        self.type = 'mhd'
        self.img_mhd = path_mhd_image + '/' + self.name1 + '/' + self.name + '/' + self.name1 + '0.png'
        print(self.img_mhd)
        img3 = QtGui.QPixmap(self.img_mhd).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img3)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_last.clicked.connect(self.last_image)

    # 下一张处理完的图片
    def next_IOUI(self):
        self.image_index = self.image_index + 1
        jpg = QtGui.QPixmap(self.file_path[self.image_index]).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

    # 上一张处理完的图片
    def last_IOUI(self):
        self.image_index = self.image_index - 1
        jpg = QtGui.QPixmap(self.file_path[self.image_index]).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

    # mha首位图片加载
    def openimage_mha(self):

        self.num = 0
        self.type = 'mha'
        self.img_mha = path_mha_image + '.' + '/' + self.name1 + '.' + '/' + self.name + '.' + '/' + self.name1 + '0.png'
        img3 = QtGui.QPixmap(self.img_mha).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img3)

        self.btn_next.clicked.connect(self.next_image)
        self.btn_last.clicked.connect(self.last_image)

    # mhd首位图片加载
    def openimage_mhd(self):

        self.num = 0
        print('self.num', self.num)
        self.type = 'mhd'
        self.img_mhd = path_mhd_image + '.' + '/' + self.name1 + '.' + '/' + self.name + '.' + '/' + self.name1 + '0.png'
        img3 = QtGui.QPixmap(self.img_mhd).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img3)

        self.btn_next.clicked.connect(self.next_image)
        self.btn_last.clicked.connect(self.last_image)

    # 下一张图片
    def next_image(self):

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

    # 上一张图片
    def last_image(self):

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

    # 获取鼠标点击位置
    def mousePressEvent(self, e):
        self.x = e.pos().x()
        self.y = e.pos().y()
        i_num = int((self.x - 30) * 1024/700)
        j_num = int((self.y - 100) * 1024/700)
        self.label_location.setText('鼠标点击位置:' + str(j_num) + ',' + str(i_num))
        self.btn_mouse.clicked.connect(self.mouse_click)
        self.btn_batch.clicked.connect(self.mouse_click_batch)

    def get_max(self, a, b):
        i = 0
        if a > b:
            i = a
        else:
            i = b
        return i

    def get_min(self, a, b):
        i = 0
        if a < b:
            i = a
        else:
            i = b
        return i

    # 鼠标点击批处理功能模块
    def mouse_click_batch(self):
        target_num = self.num
        start_num = self.get_max((target_num - 30), 1)
        end_num = self.get_min((target_num + 30), self.image_num)
        print('target_num =', target_num)
        print('start_num =', start_num)
        print('end_num =', end_num)
        total_num = end_num - start_num
        for i in range(start_num, end_num):
            print('正在处理编号{}图片，总共有{}张图片'.format(i, total_num))
            self.num = i
            self.mouse_click()
        jpg = QtGui.QPixmap(self.IOUimage).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

    # 鼠标点击处理功能模块
    def mouse_click(self):

        if self.type == 'mhd':
            print(self.name1, self.name, self.name1 + str(self.num))
            self.locationx = (self.x - 40) * 1024/700
            self.locationy = (self.y - 120) * 1024/700
            print(self.locationx, self.locationy)
            print(self.locationx, self.locationy)

            self.acc = process(str(self.name1), str(self.name), str(self.name1 + str(self.num)), int(self.locationx), int(self.locationy))
            self.IOUimage = str(path + '/' + str(self.name1 + str(self.num)) + '.png')
            print('IOUimage = ', self.IOUimage)
            jpg = QtGui.QPixmap(self.IOUimage).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)
            self.btn_IOU.clicked.connect(self.IOU)
            self.btn_volume.clicked.connect(self.Volume)
            self.btn_weight.clicked.connect(self.Weight)
            # self.btn_next.clicked.connect(self.nextIOUI)
            # self.btn_last.clicked.connect(self.lastIOUI)

    # 显示下一张处理完的图片
    def nextIOUI(self):
        self.num = self.num + 1
        self.IOUimage = str(path + str(self.name1 + str(self.num)) + '.png')
        jpg = QtGui.QPixmap(self.IOUimage).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.btn_next.clicked.connect(self.nextIOUI)
        self.btn_last.clicked.connect(self.lastIOUI)

    # 显示上一张处理完的图片
    def lastIOUI(self):
        self.num = self.num - 1
        self.IOUimage = str(path + '/' + str(self.name1 + str(self.num)) + '.png')
        jpg = QtGui.QPixmap(self.IOUimage).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.btn_next.clicked.connect(self.nextIOUI)
        self.btn_last.clicked.connect(self.lastIOUI)

    # IOU值处理
    def IOU(self):
        self.label_IOU.setText('IOU:' + str(self.acc))

    # 计算体积大小值
    def Volume(self):
        self.volume = volume_count(str(self.name1), str(self.name), int(self.locationx), int(self.locationy))
        volume_all = self.volume * (1024/798) * (1024/798) * 0.03089
        self.label_volume.setText('体积大小:' + str(int(volume_all)) + 'mm**3')

    # 计算重量大小值
    def Weight(self):
        self.weight = weight_count(str(self.name1), str(self.name), int(self.locationx), int(self.locationy))
        weight_all = self.weight * (1024/798) * (1024/798) * 1000 / 1000000 * 0.03089
        self.label_weight.setText('重量大小:' + str(int(weight_all)) + 'mg')

    # 设置主界面背景皮肤
    def use_palette(self):
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("./image/main_skin_04.jpg")))
        self.setPalette(window_pale)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
