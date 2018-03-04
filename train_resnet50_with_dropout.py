__author__ = "ben"

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

from utils.model_util import load_resnet50_with_dropout, train_model_imagedatagen
from utils.img_preprocessing import get_img_gen
import pandas as pd
import numpy as np
from utils.file_util import write_result2df_json

input_size = 512
input_channels = 3
batch_size = 15
epochs = 50
kfold = 5

df_train_all = pd.read_csv("../output/train_label.csv")
df_valid = pd.read_csv("../output/validation_label.csv")
train_path = "../input/train"
valid_path = "../input/valid"
model_save_file = "weight/resnet50_with_dropout"
result_save_dir = "../result/resnet50_with_dropout"

train_gen = get_img_gen()

def _predict(result, pred_filenames):
    converted_result = np.zeros_like(result)  # real probability after converted
    mapping = pd.read_csv("../input/mapping.txt", header=None, names=["maps"]).maps.tolist()
    print(mapping)
    for j in range(80):
        print(mapping[j])
        converted_result[:, mapping[j]] = result[:, j]
    print(converted_result.shape)
    pd.DataFrame({"filename": pred_filenames}).to_csv("../result/resnet50_with_dropout/test_filename.csv", index=None)
    np.save("../result/resnet50_with_dropout/real_test_2_result.npy", result)

    write_result2df_json(result, df_test=pd.DataFrame({"filename": pred_filenames}),
                         df_filename="../result/resnet50_with_dropout/real_test_2_result.csv",
                         json_filename="../result/resnet50_with_dropout/real_test_2_result.json")

model = load_resnet50_with_dropout(input_size=input_size)
y_full_test = []

for fold in range(kfold):
    train_dir = os.path.join("../input/train", "train_fold_"+str(fold))
    valid_dir = os.path.join("../input/valid", "valid_fold_"+str(fold))
    checkpoint_file = os.path.join("weights", "resnet50_with_dropout", "fold_"+str(fold))+".hdf5"
    len_train, len_valid = 0, 0
    for dir in os.listdir(train_dir):
        len_train += len(os.listdir(os.path.join(train_dir, dir)))
    for dir in os.listdir(valid_dir):
        len_valid += len(os.listdir(os.path.join(valid_dir, dir)))
    train_model_imagedatagen(model=model, batch_size = batch_size, checkpoint_file=checkpoint_file,
                         train_gen=train_gen[0].flow_from_directory(train_dir, shuffle=True,
                                target_size=(input_size, input_size), batch_size=batch_size),
                         valid_gen=train_gen[1].flow_from_directory(valid_dir, shuffle=True,
                                target_size=(input_size, input_size), batch_size=batch_size),
                         len_train=len_train, len_valid=len_valid, epochs=epochs)
    model.save_weights(os.path.join(model_save_file, str(fold) + ".hdf5"))
    predict_gen = train_gen[1].flow_from_directory("../input/test/test", shuffle=False, class_mode=None,
                                                   target_size=(input_size, input_size), batch_size=batch_size)
    pred_filenames = [x.split("/")[1].split(".")[0] for x in predict_gen.filenames]
    result = model.predict_generator(generator=predict_gen, steps=(len(df_train_all) - 1) // batch_size + 1, verbose=1)
    np.save(os.path.join(result_save_dir, "result_" + str(fold) + ".npy"))
    y_full_test.append(result)

p_test = y_full_test[0]
for i in range(1, kfold):
    p_test += y_full_test

p_test /= kfold

_predict(p_test, pred_filenames)
