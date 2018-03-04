import numpy as np
import pandas as pd
import json
from tqdm import tqdm

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
        json_list.append({"image_id": df_test.iloc[i, 0] + ".jpg",
                          "label_id": [int(top3_result[i, 0]), int(top3_result[i, 1]), int(top3_result[i, 2])]})
    pd.DataFrame({"image_id": df_test.iloc[:, 0].tolist(), "top1": top1, "top2": top2, "top3": top3}).\
        to_csv(df_filename, index=None)

    with open(json_filename, "w") as f:
        f.write(json.dumps(json_list, sort_keys=True))
