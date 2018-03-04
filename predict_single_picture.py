import numpy as np
from utils.model_util import *
from utils.result_util import map_result
import os
import pandas as pd
from PIL import Image as pil_img
from tqdm import tqdm

input_size = 512
input_channel = 3

# return five rectangular crops for each picture
def crop_1(img, rate=0.8):
    batch_imgs = []
    height = np.asarray(img).shape[0]
    width = np.asarray(img).shape[1]

    # upper left
    batch_imgs.append(
        np.asarray(img.crop(box=(0, 0, int(width * rate), int(height * rate))).
                   resize((input_size, input_size), resample=pil_img.BICUBIC))
    )

    # upper right
    batch_imgs.append(
        np.asarray(img.crop(box=(0, int(height * (1 - rate)), int(width * rate), height)).
                   resize((input_size, input_size), resample=pil_img.BICUBIC))
    )

    # buttom left
    batch_imgs.append(
        np.asarray(img.crop(box=(int(width * (1 - rate)), 0, width, int(height * rate))).
                   resize((input_size, input_size), resample=pil_img.BICUBIC))
    )

    # buttom right
    batch_imgs.append(
        np.asarray(img.crop(box=(int(width * (1 - rate)), int(height * (1 - rate)), width, height)).
                   resize((input_size, input_size), resample=pil_img.BICUBIC))
    )

    # center
    batch_imgs.append(
        np.asarray(img.crop(box=(
        int(width * rate / 2), int(height * rate / 2), int(width * (1 - rate / 2)), int(height * (1 - rate / 2)))).
                   resize((input_size, input_size), resample=pil_img.BICUBIC))
    )
    print(np.array(batch_imgs).shape)
    return np.array(batch_imgs)

# return a square crop with edge length equals the shorter length of origional image
def crop_2(img):
    src_array = np.asarray(img)
    height = src_array.shape[0]
    width = src_array.shape[1]
    min_len = min(height, width)
    if height > width:
        left = np.random.randint(0, height - width)
        return img.crop(box=(0, left, min_len, min_len + left))
    elif height < width:
        right = np.random.randint(0, width - height)
        return img.crop(box=(right, 0, min_len + right, min_len))
    else:
        return img

img_prefix = "../input/valid_img/"
dirs = os.listdir(img_prefix)

origin_cnt = 0
later_cnt = 0

# vgg16, vgg19, resnet50, resnet50_with_dropout, inceptionV3, inception_resnet_v2
model = load_inception_resnetv2(input_size=input_size)
real_img_lable = pd.read_csv("../output/ordered_valid_result.csv", header=0)

for dir in dirs:
    print("============================"+dir+"============================================")
    for _img in tqdm(os.listdir(os.path.join(img_prefix, dir))):
        img_name = os.path.join(img_prefix,dir,_img)
        img_id = img_name.split("_")[-1].split(".")[0]
        p_img = pil_img.open(img_name)

        img1 = np.asarray(
            np.expand_dims(p_img.resize((input_size, input_size), resample=pil_img.BICUBIC), axis=0)
        )

        threshold = 0.5  # this threshold needs to change in order to get a higher accuracy in valid data
        result = model.predict(img1)
        if result[0, np.argsort(-result)[0, 0]] < threshold:
            croped_batch = crop_1(p_img, rate=0.6)  # resized array, the two crop methods will be used together according to the accuracy
            result_1 = model.predict(croped_batch)
            mean_result = result_1[0]
            for i in range(1, result_1.shape[0]):
                mean_result += result_1[i]
            maped_mean = map_result(np.expand_dims(np.argsort(-mean_result), axis=0))
            mean_result /= result_1.shape[0]

            arg_res1 = map_result(np.argsort(-mean_result))
        else:
            arg_res = map_result(np.argsort(-result))

        if real_img_lable.loc[real_img_lable.image_id == img_id, "lable_id"].tolist()[0] in list(maped_mean[0][:3]):
            later_cnt += 1
        if real_img_lable.loc[real_img_lable.image_id == img_id, "lable_id"].tolist()[0] in list(arg_res[0][:3]):
            origin_cnt += 1
        else:
            print("=====================" + img_name + "========================================")
            print(arg_res)
            # print(arg_res1[:, :3])
            # for i in range(result_1.shape[0]):
            #     print(result_1[i][false_result[i][0]], result_1[i][false_result[i][1]], result_1[i][false_result[i][2]])
            print(map_result(np.expand_dims(np.argsort(-mean_result), axis=0)))

print("origin cnt:",origin_cnt, "later cnt:",later_cnt)
