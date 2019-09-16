import numpy as np
import os
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt

# path = r'C:\Users\dell\Desktop\2019.5.6\data_all\chenhui/chenhui1.mhd'

path_npy_mha = r'C:\Users\dell\Desktop\2019.5.6\all_npy_and_image\npy_mha'
path_npy_mhd = r'C:\Users\dell\Desktop\2019.5.6\all_npy_and_image\npy_mhd'

path_image_mha = r'C:\Users\dell\Desktop\2019.5.6\all_npy_and_image\Cv2_image_mha'
path_image_mhd = r'C:\Users\dell\Desktop\2019.5.6\all_npy_and_image\Cv2_image_mhd'
path_image_mhd1 = r'C:\Users\dell\Desktop\2019.5.6\all_npy_and_image\Cv2_image_mhd1'


def data_read(path):
    if path[len(path) - 4:len(path)] == '.mha':
        name1, name, num = data_mha(path)
    if path[len(path) - 4:len(path)] == '.mhd':
        name1, name, num = data_mhd(path)
    return name1, name, num


def data_mha(path):
    c = len(path)
    c1 = 0
    for i in range(c):
        if path[i] == "/":
            c1 = i
    name = path[c1 + 1: c - 4]
    # print(name)
    t = 0
    for i in range(len(name)):
        if '0' <= name[i] <= '9':
            t += 1

    name1 = name[:len(name) - t]
    print(name1)
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)

    if not os.path.exists(path_npy_mha + '.' + '/' + name1):
        os.makedirs(path_npy_mha + '.' + '/' + name1)

    if not os.path.exists(path_npy_mha + '.' + '/' + name1 + '.' + '/' + name):
        os.makedirs(path_npy_mha + '.' + '/' + name1 + '.' + '/' + name)

    if not os.path.exists(path_image_mha + '.' + '/' + name1):
        os.makedirs(path_image_mha + '.' + '/' + name1)

    if not os.path.exists(path_image_mha + '.' + '/' + name1 + '.' + '/' + name):
        os.makedirs(path_image_mha + '.' + '/' + name1 + '.' + '/' + name)

    for i in range((len(image))):
        img = np.squeeze(image[i])
        np.save(path_npy_mha + '.' + '/' + name1 + '.' + '/' + name + '.' + '/' + name1 + str(i) + '.npy', img)

    for i in range(80):
        # for i in range((len(image))):
        img = np.squeeze(image[i])
        for j in range(100, 900):
            for k in range(100, 900):
                if img[j][k] == 1:
                    img[j][k] = -300
        print(i)
        plt.figure(figsize=(10.24, 10.24))
        plt.imshow(img, cmap='Greys_r')
        plt.savefig(path_image_mha + '.' + '/' + name1 + '.' + '/' + name + '.' + '/' + name1 + str(i) + '.png')
        # cv2.imwrite(path_image_mha + '.' + '/' + name1 + '.' + '/' + name + '.' + '/' + name1 + str(i) + '.png', img)
        plt.cla()
    return name1, name, len(image)


def data_mhd(path):
    c = len(path)
    c1 = 0
    for i in range(c):
        if path[i] == "/":
            c1 = i
    name = path[c1 + 1: c - 4]
    print(name)
    t = 0
    for i in range(len(name)):
        if '0' <= name[i] <= '9':
            t += 1
    name1 = name[:len(name) - t]
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)

    if not os.path.exists(path_npy_mhd + '.' + '/' + name1):
        os.makedirs(path_npy_mhd + '.' + '/' + name1)

    if not os.path.exists(path_npy_mhd + '.' + '/' + name1 + '.' + '/' + name):
        os.makedirs(path_npy_mhd + '.' + '/' + name1 + '.' + '/' + name)

    if not os.path.exists(path_image_mhd + '.' + '/' + name1):
        os.makedirs(path_image_mhd + '.' + '/' + name1)

    if not os.path.exists(path_image_mhd + '.' + '/' + name1 + '.' + '/' + name):
        os.makedirs(path_image_mhd + '.' + '/' + name1 + '.' + '/' + name)

    if not os.path.exists(path_image_mhd1 + '.' + '/' + name1):
        os.makedirs(path_image_mhd1 + '.' + '/' + name1)

    if not os.path.exists(path_image_mhd1 + '.' + '/' + name1 + '.' + '/' + name):
        os.makedirs(path_image_mhd1 + '.' + '/' + name1 + '.' + '/' + name)

    for i in range((len(image))):
        img = np.squeeze(image[i])
        np.save(path_npy_mhd + '.' + '/' + name1 + '.' + '/' + name + '.' + '/' + name1 + str(i) + '.npy', img)

    for i in range(80):
        img = np.squeeze(image[i])
        print(i)
        plt.figure(figsize=(10.24, 10.24))
        plt.imshow(img, cmap='Greys_r')
        plt.savefig(path_image_mhd + '.' + '/' + name1 + '.' + '/' + name + '.' + '/' + name1 + str(i) + '.png')
        plt.cla()

    for i in range(80):
        print(i)
        img = np.squeeze(image[i])
        plt.figure(figsize=(10.24, 10.24))
        plt.imshow(img)
        plt.savefig(path_image_mhd1 + '.' + '/' + name1 + '.' + '/' + name + '.' + '/' + name1 + str(i) + '.png')

        plt.cla()

    return name1, name, len(image)

# c, d , e =data_read(str("'" + 'C:/Users/dell/Desktop/2019.5.6/data_all/chenhui/chenhui1.mhd'))
# print(c, d, e)
