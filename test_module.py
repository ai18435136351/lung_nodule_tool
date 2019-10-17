import os
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from outline_gen import process
from data_read import data_read
from volume import volume_count, weight_count
from maker import generate

filepath = 'D:/Python/lung_nodule_tool-master/data_mha_mhd/mhd_png/chenhui/chenhui1/chenhui0.png'
file_dir = filepath.split('/')
print(file_dir[5])
print(file_dir[6])
