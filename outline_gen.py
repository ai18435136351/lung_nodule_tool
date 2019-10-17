import cv2
import numpy as np
import os
from region_growth import region_growths
from canny_edge import edge_demo
from ostu import ostu_process
from outline_supple import filling_internel
from IOU import acc
import matplotlib.pyplot as plt

# path = r'data_mha_mhd\mhd_png/chenhui/chenhui1/chenhui0'

path_data_mhd = r'data_mha_mhd/mhd_png'
path_data = r'data_mha_mhd/mhd1_png'
path_mask = r'data_mha_mhd/mha_png'

path_equlization = './img_process' + './hist_equalization'
path_region_growth = './img_process' + './region_growth'
path_equal_slice = './img_process' + './equal_slice'
path_Gauss_ostu = './img_process' + './Gauss_ostu'
path_canny = './img_process' + './canny'
path_supple_region = './img_process' + './supple_region'
path_result = './img_process' + './process'
path_equal_slice_canny = './img_process' + './equal_slice_canny'
path_slice_intrenel = './img_process' + './slice_intrenel'

# path_data_npy = 'C:\Users\wzt\Desktop\2019.5.6\all_npy_and_image\npy_mha'
path_npy_data = 'data_mha_mhd/mhd_npy'

if not os.path.exists(path_equlization):
    os.makedirs(path_equlization)
if not os.path.exists(path_region_growth):
    os.makedirs(path_region_growth)
if not os.path.exists(path_equal_slice):
    os.makedirs(path_equal_slice)
if not os.path.exists(path_Gauss_ostu):
    os.makedirs(path_Gauss_ostu)
if not os.path.exists(path_canny):
    os.makedirs(path_canny)
if not os.path.exists(path_supple_region):
    os.makedirs(path_supple_region)
if not os.path.exists(path_result):
    os.makedirs(path_result)


def cross(img11, x, y):                                                       # 内部填充
    nums = [0, 0, 0, 0]
    for i in range(81 - x):  # 横
        if img11[x + i][y] == 255:
            nums[0] = nums[0] + 1
            break
    for i in range(x):
        if img11[x - i - 1][y] == 255:
            nums[0] = nums[0] + 1
            break
    for i in range(y):  # 竖
        if img11[x][y - i - 1] == 255:
            nums[1] = nums[1] + 1
            break
    for i in range(81 - y):
        if img11[x][y + i] == 255:
            nums[1] = nums[1] + 1
            break
    for i in range(81 - 1 - x):
        if y + i + 1 >= 70:
            break
        if img11[x + i + 1][y + i + 1] == 255:  # 斜1
            nums[2] = nums[2] + 1
            break
    for i in range(x):
        if y - i - 1 <= 0:
            break
        if img11[x - i - 1][y - i - 1] == 255:  # 斜1
            nums[2] = nums[2] + 1
            break
    for i in range(81 - 1 - x):
        if y - i - 1 <= 0:
            break
        if img11[x + i + 1][y - i - 1] == 255:
            nums[3] = nums[3] + 1
            break
    for i in range(x):
        if y + i + 1 >= 81 - 1:
            break
        if img11[x - i - 1][y + i + 1] == 255:  # 斜1
            nums[3] = nums[3] + 1
            break
    num = 0
    for i in range(4):
        if nums[i] == 2:
            num = num + 1
    if num >= 2:
        return 1
    else:
        return 0
# 均衡化
def equalization(file1, file2, image_names):              # 均衡化
    img_equal1 = np.array(cv2.imread(path_data + '.' + '/' + file1 + './' + file2 + '.' + '/' + image_names + '.png'))
    l, w, h = img_equal1.shape
    hist, bins = np.histogram(img_equal1.flatten(), 256, [0, 256])
    cdf = hist.cumsum()  # 实现叠加
    equalizations = []
    for n in range(256):
        equalizations.append(int(255*(cdf[n] / (h*w*l))+0.5))
    for k in range(h):
        for i in range(l):
            for j in range(w):
                m = img_equal1[i][j][k]
                img_equal1[i][j][k] = equalizations[m]
    cv2.imwrite(path_equlization + '.' + '/' + image_names + '.png', img_equal1)
    return img_equal1


def process(file1, file2, image_names, y, x):
    # print('处理过程')
    img = np.array(cv2.imread(path_data + '.' + '/' + file1 + '.' + '/' + file2 + '.' + '/' + image_names + '.png', 0))
    # img = np.array(cv2.imread(path_data + '/' + file1 + '/' + file2 + '/' + image_names + '.png', 0))  # 原图单层
    print(img.shape)
    equalization(file1, file2, image_names)
    img_equal = np.array(cv2.imread(path_equlization + '.' + '/' + image_names + '.png', 0))          # 均衡化二维
    # img_equal = np.array(cv2.imread(path_equlization + '/' + image_names + '.png', 0))
    print(img_equal.shape)
    img_mask = np.array(cv2.imread(path_mask + '.' + '/' + file1 + '.' + '/' + file2 + '.' + '/' + image_names + '.png', 0))
    # img_mask = np.array(cv2.imread(path_mask + '/' + file1 + '/' + file2 + '/' + image_names + '.png', 0))
    print(img_mask.shape)
    # locationx = []
    # locationy = []
    #
    # for i in range(150, 900):
    #     for j in range(150, 900):
    #         if img_mask[i][j] > 0:
    #             locationx.append(i)
    #             locationy.append(j)
    #
    # x = int((max(locationx) + min(locationx)) / 2)
    # y = int((max(locationy) + min(locationy)) / 2)
    # print(x, y)

    max_num = 77
    min_num = 50
    x1 = 81
    x2 = 40

    img1 = np.zeros((x1, x1))  # 原图 切块
    img2 = np.zeros((x1, x1))  # 均衡化图 切块

    for i in range(x1):
        for j in range(x1):
            img1[i][j] = img[i + x - x2][j + y - x2]
            img2[i][j] = img_equal[i + x - x2][j + y - x2]

    img3 = region_growths(img1, image_names, 2)                                    # 均衡化图 切块 区域增长
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img3 = cv2.dilate(img3, kernel)  # 原图切块增长后 膨胀
    img3 = cv2.dilate(img3, kernel)  # 膨胀
    img3 = cv2.dilate(img3, kernel)  # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img3 = cv2.dilate(img3, kernel)  # 膨胀
    img3 = cv2.dilate(img3, kernel)  # 膨胀
    img3 = cv2.dilate(img3, kernel)  # 膨胀

    cv2.imwrite(path_region_growth + '.' + '/' + image_names + '.png', img3)        # 区域生长图保存
    for i in range(x1):
        for j in range(x1):
            if img2[i][j] > max_num:
                img2[i][j] = 0
            elif img2[i][j] < min_num:
                img2[i][j] = 0
            else:
                img2[i][j] = 70

    cv2.imwrite(path_equal_slice + '.' + '/' + image_names + '.png', img2)        # 均衡化图阈值处理 转化为三维
    img4 = cv2.imread(path_equal_slice + '.' + '/' + image_names + '.png', 0)     # img4 为均衡化图 img2 只有两个像素
    path = path_data + '.' + '/' + file1 + '.' + '/' + file2 + '.' + '/' + image_names
    img4 = ostu_process(path, img4)
    cv2.imwrite(path_Gauss_ostu + '.' + '/' + image_names + '.png', img4)

    img4 = cv2.imread(path_Gauss_ostu + '.' + '/' + image_names + '.png')
    img4 = edge_demo(img4, image_names)  # 均衡化切块后canny算子

    cv2.imwrite(path_canny + '.' + '/' + image_names + '.png', img4)        # 均衡化图阈值处理 转化为三维

    for i in range(x1):  # img4 区域内处理
        for j in range(x1):
            if img3[i][j] == 100:
                img4[i][j] = img4[i][j]
            else:
                img4[i][j] = 0
    #
    img4 = filling_internel(img4, image_names)  # 内部填充后轮廓
    flag = np.zeros((x1, x1))
    flag1 = np.zeros((x1, x1))

    for i in range(x1):
        for j in range(x1):
            flag[i][j] = cross(img4, i, j)
    for i in range(x1):
        for j in range(x1):
            if flag[i][j] == 1:
                img4[i][j] = 255
    for i in range(x1):
        for j in range(x1):
            flag1[i][j] = cross(img4, i, j)
    for i in range(x1):
        for j in range(x1):
            if flag1[i][j] == 1:
                img4[i][j] = 255

    img4 = region_growths(img4, image_names, 2)  # 原图切块 区域增长

    cv2.imwrite(path_supple_region + '.' + '/' + image_names + '.png', img4)
    img_edge = cv2.imread(path_equal_slice_canny + '.' + '/' + image_names + '.png', 0)

    img6 = np.array(cv2.imread(path_data_mhd + '.' + '/' + file1 + '.' + '/' + file2 + '.' + '/' + image_names + '.png', 0))

    for i in range(x1):
        for j in range(x1):
            if img_edge[i][j] != 0:
                img_equal[i + x - x2][j + y - x2] = img_equal[i + x - x2][j + y - x2] + 100
                img6[i + x - x2][j + y - x2] = img_equal[i + x - x2][j + y - x2] + 300

    cv2.imwrite(path_result + '.' + '/' + image_names + '.png', img6)
    img5 = np.array(cv2.imread(path_slice_intrenel + '.' + '/' + image_names + '.png', 0))    # 图片生成切块
    acc1 = acc(img5, img_mask, y, x)
    print(acc1)
    return acc1

def area(file1, file2, data_name, x, y):
    process(file1, file2, data_name, x, y)
    img_area =  np.array(cv2.imread(path_slice_intrenel + '.' + '/' + data_name + '.png', 0))
    t = 0
    for i in range(81):
        for j in range(81):
            if img_area[i][j] > 0:
                t += 1
    return t

def weight(file1, file2, data_name, x, y):
    img_weight = np.array(cv2.imread(path_slice_intrenel + '.' + '/' + data_name + '.png', 0))
    npy_data = np.load(path_npy_data + './' + file1 + './' + file2 + './' + data_name + '.npy')
    t1 = 0
    t2 = 0
    img = np.zeros((81, 81))
    for i in range(81):
        for j in range(81):
            if img_weight[i][j] > 0:
                HU = npy_data[int((i + y - 130 - 40)*1024/798)][int((j + x - 122 - 40)*1024/798)]
                Den = HU + 1000
                t1 = t1 + Den
                t2 += 1
            # print(npy_data[i + 604 - 40][j + 725 - 40])
            # img[i][j] = npy_data[int((i + y - 130 - 40)*1024/798)][int((j + x - 122 - 40)*1024/798)]
    # print(t1)
    return t1
# weight('chenhui', 'chenhui1', 'chenhui37', 604, 725)


# process('chenhui', 'chenhui1', 'chenhui35', 604, 725)

# area1 = area('chenhui', 'chenhui1', 'chenhui35')
# print(area1)
