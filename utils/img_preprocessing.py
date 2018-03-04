# -*- coding: utf-8 -*

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import cv2
import math

# keras.preprocessing.image源码简单修改，需要在keras.preprocessing.image.load_img函数中进行rescale之前先进行crop
def preprocess_pointer(src):
    src_array = np.asarray(src)
    rate = np.random.randint(60,85) / 100

    # rate = 0.6
    height = src_array.shape[0]
    width = src_array.shape[1]

#    height_random = np.random.randint(0, int(height*(1-rate)))
#    width_random = np.random.randint(0, int(width*(1-rate)))
#    print(width_random, height_random, width_random+int(width*rate), height_random+int(height*rate))
#    left, upper, right, lower
#    return src.crop(box=(width_random, height_random, width_random+int(width*rate), height_random+int(height*rate)))
    min_len = min(height, width)
    if height > width:
        left = np.random.randint(0, height-width)
        return src.crop(box=(0, left, min_len, min_len+left))
    elif height < width:
        right = np.random.randint(0, width-height)
        return src.crop(box=(right, 0, min_len+right, min_len))
    else:
        return src

def get_img_gen():

    ## crop操作修改 keras.preprocessing.image, 在里面加上preprocess_pointer函数。
    datagen1 = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
    )
    datagen2 = ImageDataGenerator()

    # set data augmentation parameters here
    datagen3 = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=.25,
        height_shift_range=.25,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect",
        channel_shift_range=50
    )

    # normalization neccessary for correct image input to VGG16
    # datagen4.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

    # no data augmentation for validation and test set
    # validgen = ImageDataGenerator(rescale=1., featurewise_center=True)
    # validgen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

    return datagen1, datagen2, datagen3


def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
        mask = shift(mask, wshift, hshift)
    return img, mask


def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_gray(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return img


def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img


def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img


def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
    return img


def brightness_augment(img, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb


def color_normalize(img, meanstd):
    """
    img为输入图像，meanstd为map{"mean": [m1, m2, m3], std: [s1, s2, s3]}
    :param img:
    :param meanstd:
    :return:
    """
    dst = img
    for i in range(dst.shape[2]):
        dst[:, :, i] -= meanstd.mean[i]
        dst[:, :, i] /= meanstd.std[i]
    return dst


def scale(img, size, interpolation=cv2.INTER_CUBIC):
    """
    用双立方插值法将原图像按照等比例缩放，小边的长度为size
    :param img:
    :param size:
    :param interpolation:
    :return:
    """
    h, w = img.shape[0], img.shape[1]
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        return cv2.resize(img, dsize=(size, h/w * size), interpolation=interpolation)
    else:
        return cv2.resize(img, dsize=(w/h * size, size), interpolation=interpolation)

def center_crop(img, size):
    """
    居中截取
    :param img:
    :param size:
    :return:
    """
    w1 = math.ceil((img.shape[1]-size)/2)
    h1 = math.ceil((img.shape[0]-size)/2)
    return img[w1:w1+size, h1:h1+size, :]

def random_crop(img, size, padding=0):
    """
    随机截取一个(size, size)大小的图片
    :param img:
    :param size:
    :param padding:
    :return:
    """
    if padding > 0:
        temp = np.zeros((img.shape[0]+2*padding, img.shape[1]+2*padding, 3))
        temp[padding+1:img.shape[0], padding+1:img.shape[1], :] = img
        img = temp

    h, w = img.shape[0], img.shape[1]
    if h == size and h == size:
        return img

    x1, y1 = np.random.randint(0, w-size), np.random.randint(0, h-size)
    output = img[y1:y1+size, x1:x1+size, :]
    return output

def ten_crop(img, size):
    """
    截取4个角落和正中央的一部分，然后水平翻转和再次截取，每张图片截取10次返回
    :param img:
    :param size:
    :return:
    """
    cen_crop = center_crop(img, size)
    w, h = img.shape[2], img.shape[2]
    output = []
    for _img in [img, cv2.flip(img, flipCode=1)]:
        output.append(_img)
        output.append(_img[0:size, 0:size, :])
        output.append(_img[w-size:w, 0:size, :])
        output.append(_img[0:size, h-size:h, :])
        output.append(_img[w-size:w, h-size:h, :])

    return output.append(cen_crop)


def randomScale(img, minSize, maxSize):
    """
    保留小边为targetSz，大边变为targetSz * 倍数，即等比例缩放
    :param img:
    :param minSize:
    :param maxSize:
    :return:
    """
    w, h = img.shape[2], img.shape[1]
    targetSz = np.random.randint(minSize, maxSize)
    targetW, targetH = targetSz, targetSz
    if w < h:
        targetH = np.round(h / w * targetW)
    else:
        targetW = np.round(w / h * targetH)

    return cv2.resize(img, dsize=(targetW, targetH), interpolation=cv2.INTER_CUBIC)
