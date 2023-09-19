
from glob import glob

import numpy as np
import pandas as pd


path = "HPC_Data/0_zf_0_b/predictions"
pred_paths = glob(path + '/*')
pred_list = []
for i in range(pred_paths.__len__()):
    pred_list.append(np.load(pred_paths[i]))

pred_df = pd.DataFrame(np.concatenate(pred_list, axis=0 ), columns=['pred_value'])
total_preds = 0   
for pred in pred_list:
    print(pred)
    total_preds += len(pred)

print ("total_preds:", total_preds)
print ("len(pred_list):", len(pred_list))