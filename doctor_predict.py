import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
import glob
import pandas as pd

import keras
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras import backend as K
import tensorflow as tf

working_path = "./image/"

path = '/media/xxw/code/wzt/2019.5.6/doctor_train_image/lung/chenhui35.png'
# K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same' )(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same' )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=3.0e-4), loss=bce_dice_loss, metrics=[dice_coef, mean_iou])
    # model.compile(optimizer=Adam(lr=1.0e-4), loss = 'binary_crossentropy', metrics=[dice_coef])

    return model


# images = np.load('./testImages.npy')
# print(images.shape)


def normalize_hu(image):
    #将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def get_pixels_hu_by_simpleitk(image_path):

    #-----读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)
    # reader = SimpleITK.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir) # 参数，给定当前路劲
    # reader.SetFileNames(dicom_names)
    # image = reader.Execute()
    # img_array = SimpleITK.GetArrayFromImage(image)

    #------如果读取单个 dicom文件
    # image = SimpleITK.ReadImage(image_path) #参数，给定某个dicom的文件路径
    # image_array = SimpleITK.GetArrayFromImage(image) # z, y, x
    image_array = np.load(image_path)
    image_array[image_array == -2000] = 0
    return image_array

# count = 0
# for i in tqdm(range(images.shape[0])):
#     if count>5:
#         break
#     images[i, 0, :, :][images[i, 0, :, :] == -2000] = 0
#     images[i, 0, :, :] = normalize_hu(images[i, 0, :, :])
#     pre = model.predict(np.expand_dims(images[i].transpose(1, 2, 0), 0))[0]
#     print(pre[:10,:10])
#     cv2.imwrite('./submit/predict_'+str(i) + '.png', ((pre>0.5)*255).astype(np.uint8))
#     count += 1
# 数据输入网络之前先进行预处理
'''
def prepare_image_for_net(img):
    img = img.astype(np.float)
    img /= 255.
    if len(img.shape) == 3:
        img = img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])
    else:
        # img = img.reshape(1, img.shape[0], img.shape[1], 1)
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 3)
    return img

images = []
img = cv2.imread('./unet_input_img.png', cv2.IMREAD_GRAYSCALE)
img = np.array(img.resize(512, 512))
images.append(img)
for index, img in enumerate(images):
    img = prepare_image_for_net(img)
    images[index] = img
images3d = np.vstack(images)
y_pred = model.predict(np.expand_dims(images3d, 1), batch_size=1)
print(len(y_pred))
count = 0
for y in y_pred:
    y *= 255.
    y = y.reshape((y.shape[0], y.shape[1])).astype(np.uint8)
    cv2.imwrite('./unet_result.png', y)
    count += 1
'''
'''
img = cv2.imread('./unet_input_img.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
img = img.reshape(1, img.shape[0], img.shape[1], 1)
print(img.shape)
img3d = np.vstack(img)
print(img3d.shape)

'''
'''
image = np.load("train_data_doctor.npy")
paths = glob.glob(os.path.join('./dataall', '*.png'))
count=0
for i, j in enumerate(tqdm(paths)):
    if count > 3:
        break
    # img = np.expand_dims(((image[i,:,:,:])*255).astype(np.uint8), 2)
    
    img = model.predict(np.expand_dims(cv2.imread(j), 0))[0]
    print(img.shape)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = ((img_gray)*255).astype(np.uint8)
    cv2.imwrite('./submit/predict_'+str(i)+'.png', img)
    count += 1
'''

def compute_iou(gt_mask, predict_mask):
    # 计算测试数据的iou
    A = np.squeeze(gt_mask)
    B = np.squeeze(predict_mask)
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)

    iou = np.sum(intersection>0) / np.sum(union>0)
    return iou


def predict(path):
    # if not os.path.exists('./doctor_submit'):
    #     os.makedirs('./doctor_submit')
    PATH = './2019-06-05-23-17-03-021769'
    doctor_train_images = './doctor_train_image/lung'
    doctor_train_masks = './doctor_train_image/lungmask'
    model = get_unet()
    model.load_weights(os.path.join(PATH+os.sep+'checkpoint', 'linknet_bce_dice_loss_100_epochs.hdf5') )
    paths = glob.glob(os.path.join(doctor_train_images, '*.png'))
    count = 0
    mIOU = [0.0 for i in np.arange(0.1, 1.0, 0.05)]
#for path in tqdm(paths):
    #if count>5:
     #   break
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img_array = get_pixels_hu_by_simpleitk(i)
    # img_array = normalize_hu(img_array)
    img_normal = img / 255.
    img_normal = model.predict(np.expand_dims(np.expand_dims(img_normal, 0), 3))[0] #如果不加 [0], 则print(pre.shape) -> (1, 512, 512, 1)
    cv2.imwrite(PATH +os.sep+'predict_image'+os.sep+path.split(os.sep)[-1], (img_normal*255.).astype(np.uint8))
    print(PATH +os.sep+'predict_image'+os.sep+path.split(os.sep)[-1])
    cv2.imwrite(PATH +os.sep+'predict_image'+os.sep+ (path.split(os.sep)[-1]).split('.')[0] +'_threshold.png', ((img_normal>0.5)*255.).astype(np.uint8))
    
    # ------------------------------------------------
    # 在使用阈值预测的mask上，对原图画上预测的结节轮廓
    threshold_mask = ((img_normal>0.5)*255.).astype(np.uint8)
    
    sft_i = [-1,-1,0,1,1,1,0,-1]
    sft_j = [0,1,1,1,0,-1,-1,-1]
    
    def isIn(i, j):
        if(i>=0 and i<img_rows and j>=0 and j<img_cols):
            return True
        return False

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            flag = True # 是否center周围的值全与center相等（若全相等，则不是mask的边界）
            for k in range(8):
                ni = i + sft_i[k]
                nj = j + sft_j[k]
                if(isIn(ni,nj) and threshold_mask[ni][nj]!=threshold_mask[i][j]):
                    flag = False
                    break
                
            if not flag:
                img[i][j] = 255
    
    # -------------------------------------------------
    # 在真是标注的mask上，对原图画上预测的结节轮廓
    gt_mask = cv2.imread(os.path.join(doctor_train_masks, path.split(os.sep)[-1]), cv2.IMREAD_GRAYSCALE)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            flag = True # 是否center周围的值全与center相等（若全相等，则不是mask的边界）
            for k in range(8):
                ni = i + sft_i[k]
                nj = j + sft_j[k]
                if(isIn(ni,nj) and gt_mask[ni][nj]!=gt_mask[i][j]):
                    flag = False
                    break
                
            if not flag:
                img[i][j] = 0
    # -------------------------------------------------

    cv2.imwrite(PATH +os.sep+'predict_image'+os.sep+ (path.split(os.sep)[-1]).split('.')[0] +'_original.png', img)
    
    for i, t in enumerate(np.arange(0.1, 1.0, 0.05)):
        mIOU[i] += compute_iou(gt_mask, (img_normal>t))
    # count += 1
    #for i, j in enumerate(mIOU):
     #   mIOU[i] = j / len(paths)
    # mIOU = mIOU / 6
    #print('mIOU over all test set in (0.1-0.95):', mIOU)
        
def dice_np(y_true, y_pred):
    smooth = 1.0
    t = 0
    y_true_f = (y_true.reshape((1,-1))>t)
    y_pred_f = (y_pred.reshape((1,-1))>t)
    intersection = np.sum(y_true_f * y_pred_f)
    
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) )



def compute_iou_dice():
    pre_mask = glob.glob(os.path.join('./2019-06-05-23-17-03-021769/predict_image/post_mask','*.png'))
    df = dict()
    df['image'] = []
    df['iou'] = []
    df['dice'] = []
    for i in tqdm(pre_mask):
        pre = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

        mask_name =  ((i.split(os.sep)[-1]).split('.')[0]).split('_')[0]+'.'+(i.split(os.sep)[-1]).split('.')[-1]
        # print(mask_name)
        label = cv2.imread(os.path.join('./doctor_train_image/lungmask', mask_name ), cv2.IMREAD_GRAYSCALE)
        iou = compute_iou(label, pre )
        dice = dice_np(label, pre)
        df['image'].append(mask_name)
        df['iou'].append(iou)
        df['dice'].append(dice)
    df = pd.DataFrame(df).sort_index(axis=1)
    df.to_csv('post_iou.csv', index = False, encoding='utf-8')

def single_compute_iou_dice():
    # model = get_unet()
    # model.load_weights(os.path.join(path+os.sep+'checkpoint', 'linknet_bce_dice_loss_100_epochs.hdf5') )
    pre_path = os.path.join('./2019-06-05-23-17-03-021769/predict_image/pre_mask', )
    label_path = ''
    pre = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(os.path.join('./doctor_train_image/lungmask', label_path ), cv2.IMREAD_GRAYSCALE)
    iou = compute_iou(pre, label)
    print(label_path, iou)

#predict(path)

#if __name__ == '__main__':
    
    #predict() # 使用训练后的模型预测图像，生成mask
    
    #compute_iou_dice() # 连续计算每张图像的IoU和dice
    # single_compute_iou_dice() # 单次计算每张图像的IoU和dice
