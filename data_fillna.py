# coding:utf-8

import pandas as pd
import warnings
import sys
import numpy as np
import gc

sys_new = reload(sys)
sys_new.setdefaultencoding('utf-8')
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    data = pd.read_csv(r'D:/kaggle/health/test_set_1.csv', encoding='utf-8')

    data_1 = data['2403']
    data_2 = data['2404']
    data_3 = data['2405']
    # data1 = pd.concat([data_1, data_2])
    # data1.to_csv(r'D:/kaggle/health/temp_data/vid_0102.csv', index=False, sep=',', encoding='utf-8')
    data_1.to_csv(r'D:/kaggle/health/temp_data/2403_3.csv', sep=',', encoding='utf-8')
    data_2.to_csv(r'D:/kaggle/health/temp_data/2404_3.csv', sep=',', encoding='utf-8')
    data_2.to_csv(r'D:/kaggle/health/temp_data/2405_3.csv', sep=',', encoding='utf-8')
    # part1 = pd.read_csv(r'D:/kaggle/health/temp_data/vid.csv', encoding='utf-8')
    # part2 = pd.read_csv(r'D:/kaggle/health/temp_data/0102.csv', encoding='utf-8')

    # merge = pd.concat([part1, part2])
    # del part1
    # del part2
    # gc.collect()
    # merge.to_csv(r'D:/kaggle/health/temp_data/vid_0102.csv', sep=',', encoding='utf-8')
    # print part2.shape (38198, 1)
    # print part1.shape (38198, 1)
