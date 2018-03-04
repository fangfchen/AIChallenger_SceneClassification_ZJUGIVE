__author__ = "ben"

import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json
from utils.img_preprocessing import get_img_gen

test_dir = os.path.join("..", "input", "test_b", "test_b")

input_size = 512
batch_size = 32
# model = load_inception_resnetv2(512)

mapping = pd.read_csv("../input/mapping.txt",header=None, names=["maps"]).maps.tolist()

def write_result2df_json(result, df_test, df_filename, json_filename):
    if result.shape[0] == 1:
        result = result[0]
    top3_result = np.argsort(-result)
    top1 = []
    top2 = []
    top3 = []
    json_list = []
    for i in tqdm(range(len(df_test))):
        top1.append(int(top3_result[i, 0]))
        top2.append(int(top3_result[i, 1]))
        top3.append(int(top3_result[i, 2]))
        json_list.append({"image_id": df_test.iloc[i, 0],
                          "label_id": [int(top3_result[i, 0]), int(top3_result[i, 1]), int(top3_result[i, 2])]})
    pd.DataFrame({"image_id": df_test.iloc[:, 0].tolist(), "top1": top1, "top2": top2, "top3": top3}).\
        to_csv(df_filename, index=None)

    with open(json_filename, "w") as f:
        f.write(json.dumps(json_list, sort_keys=True))


if __name__ == "__main__":

    # copy all the result file into one folder named result/total_result
    result_dirs = "../result/total_result"
    total_result = []
    predict_gen = get_img_gen()[1].\
        flow_from_directory("../input/test/test", shuffle=False, class_mode=None,
                            target_size=(input_size, input_size), batch_size=batch_size)
    for result_filename in os.listdir(result_dirs):
        total_result.append(np.load(os.path.join(result_dirs, result_filename)))
    p_test = total_result[0]
    for i in range(1, len(total_result)):
        p_test += total_result[i]

    p_test /= len(total_result)
    filenames = [x.split("/")[1].split(".")[0] for x in predict_gen.filenames]
    df_result_file = "../result/test/fianl_submission.csv"
    josn_result_file = "../result/test/final_submission.json"
    df_test = pd.DataFrame({"image_id": filenames})

    write_result2df_json(p_test, df_test=df_test, df_filename=df_result_file,
                         json_filename=josn_result_file)  # save result to file
