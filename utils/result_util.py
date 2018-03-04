import numpy as np
import pandas as pd

def get_top3_acc(pred_result, is_train):
    if not is_train:
        df_true = pd.read_csv("../output/valid_label.csv", header=0)  # ordered valid labels with filename order ImageDataGenerator().filenames
    else:
        df_true = pd.read_csv("../output/train_label.csv", header=0)  # ordered train labels with filename order ImageDataGenerator().filenames
    result_args = np.argsort(-pred_result)
    pred = pd.DataFrame(result_args)
    cnt = 0
    print(len(pred))
    print(len(df_true))
    for i in range(len(pred)):
        if df_true.iloc[i].lable_id.tolist() in pred.iloc[i, :3].tolist():
           cnt += 1
    return 1.0 * cnt / len(pred)

def map_result(result):
    li = pd.read_csv("../input/mapping.txt", header=None, names=["mapping"]).mapping.tolist()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i][j] = int(li[result[i][j]])
    return result