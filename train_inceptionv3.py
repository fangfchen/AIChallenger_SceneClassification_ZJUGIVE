__author__ = "ben"

from utils.model_util import train_model_imagedatagen
from utils.model_util import load_inceptionV3_model
from utils.img_preprocessing import get_img_gen
import pandas as pd
import numpy as np
import os
from keras.optimizers import Adam, SGD
from utils.file_util import write_result2df_json
from keras.utils.training_utils import multi_gpu_model

input_size = 512
input_channels = 3
batch_size = 15
epochs = 50
gpus = 3
kfold = 5  # five fold cross validation

df_train_all = pd.read_csv("../output/filtered_train.csv")
df_valid = pd.read_csv("../output/validation_label.csv")
df_test = pd.read_csv("../output/sample_submission.csv")
model_save_file = "weight/inceptionV3"
result_save_dir = "../result/inceptionV3"

train_gen = get_img_gen()

def _predict(result, pred_filenames):
    converted_result = np.zeros_like(result)  # real probability after converted
    mapping = pd.read_csv("../input/mapping.txt", header=None, names=["maps"]).maps.tolist()
    print(mapping)
    for j in range(80):
        print(mapping[j])
        converted_result[:, mapping[j]] = result[:, j]
    print(converted_result.shape)
    pd.DataFrame({"filename": pred_filenames}).to_csv("../result/inceptionV3/test_filename.csv", index=None)
    np.save("../result/inceptionV3/real_test_2_result.npy", result)

    write_result2df_json(result, df_test=pd.DataFrame({"filename": pred_filenames}),
                         df_filename="../result/inceptionV3/real_test_2_result.csv",
                         json_filename="../result/inceptionV3/real_test_2_result.json")

model = load_inceptionV3_model(input_size=input_size)

adam_optimizer = Adam(lr=5*1e-4)
sgd_optimizer = SGD(lr=1e-4, decay=1e-6, momentum=0.8)
y_full_test = []

for fold in range(kfold):
    train_dir = os.path.join("../input/train", "train_fold_"+str(fold))
    valid_dir = os.path.join("../input/valid", "valid_fold_"+str(fold))
    checkpoint_file = os.path.join("weights", "inceptionV3", "fold_"+str(fold))+".hdf5"
    len_train, len_valid = 0, 0
    for dir in os.listdir(train_dir):
        len_train += len(os.listdir(os.path.join(train_dir, dir)))
    for dir in os.listdir(valid_dir):
        len_valid += len(os.listdir(os.path.join(valid_dir, dir)))
    train_model_imagedatagen(model=multi_gpu_model(model, gpus=gpus), batch_size = batch_size,
                             checkpoint_file=checkpoint_file,
                         train_gen=train_gen[0].flow_from_directory(train_dir, shuffle=True,
                                target_size=(input_size, input_size), batch_size=batch_size),
                         valid_gen=train_gen[1].flow_from_directory(valid_dir, shuffle=True,
                                target_size=(input_size, input_size), batch_size=batch_size),
                         len_train=len(df_train_all), len_valid=len(df_valid), epochs=epochs, optimizer=sgd_optimizer)
    model.save_weights(os.path.join(model_save_file, str(fold)+".hdf5"))
    predict_gen = train_gen[1].flow_from_directory("../input/test/test", shuffle=False, class_mode=None,
                                                   target_size=(input_size, input_size), batch_size=batch_size)
    pred_filenames = [x.split("/")[1].split(".")[0] for x in predict_gen.filenames]
    result = model.predict_generator(generator=predict_gen, steps=(len(df_train_all) - 1) // batch_size + 1, verbose=1)
    np.save(os.path.join(result_save_dir, "result_"+str(fold)+".npy"))
    y_full_test.append(result)

p_test = y_full_test[0]
for i in range(1, kfold):
    p_test += y_full_test

p_test /= kfold

_predict(p_test, pred_filenames)
