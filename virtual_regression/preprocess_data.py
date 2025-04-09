import pandas as pd
import numpy as np
import glob
import cv2
import os
import json

inputPath = "data/allDatas_regression.txt"

# with open(inputPath) as f:
#     lines = f.readlines()
#     for line in lines:
#         new_line = line.strip().split('|')
#         new_line = ' '.join(new_line)
#         with open("/home/jcl3689/YXW/Zhang/virtual_classification/data/format_allDatas.txt", 'a') as ff:
#             ff.write(new_line)
#             ff.write('\n')

with open(inputPath) as f:
    lines = f.readlines()
    for line in lines:
        data = []
        new_line = line.strip().split('|')
        for i, ds in enumerate(new_line):
            if i == 1 or i == 2 or i == 4 or i == 5:
                new_ds = json.loads(ds.strip())
                for d in new_ds:
                    data.append(str(d))
            else:
                data.append(ds.strip())

        data = ','.join(data)
        # print(data)
        with open("data/format_each_allDatas_regression.txt", 'a') as ff:
            ff.write(data)
            ff.write('\n')