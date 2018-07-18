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
    # 用test比较快
    # 而且如果test中是数值，则train也是数值的，一样处理

    data_1 = data['1474']
    data_1.to_csv(r'D:/kaggle/health/col/1474.csv', sep=',', encoding='utf-8')
    del data_1
    gc.collect()

    data_2 = data['669006']
    data_2.to_csv(r'D:/kaggle/health/col/669006.csv', sep=',', encoding='utf-8')
    del data_2
    gc.collect()

    data_3 = data['300017']
    data_3.to_csv(r'D:/kaggle/health/col/300017.csv', sep=',', encoding='utf-8')
    del data_3
    gc.collect()

    data_4 = data['100014']
    data_4.to_csv(r'D:/kaggle/health/col/100014.csv', sep=',', encoding='utf-8')
    del data_4
    gc.collect()

    data_5 = data['809009']
    data_5.to_csv(r'D:/kaggle/health/col/809009.csv', sep=',', encoding='utf-8')
    del data_5
    gc.collect()

    data_6 = data['1112']
    data_6.to_csv(r'D:/kaggle/health/col/1112.csv', sep=',', encoding='utf-8')
    del data_6
    gc.collect()

    data_7 = data['2386']
    data_7.to_csv(r'D:/kaggle/health/col/2386.csv', sep=',', encoding='utf-8')
    del data_7
    gc.collect()

    data_8 = data['100013']
    data_8.to_csv(r'D:/kaggle/health/col/100013.csv', sep=',', encoding='utf-8')
    del data_8
    gc.collect()

    data_9 = data['0415']
    data_9.to_csv(r'D:/kaggle/health/col/0415.csv', sep=',', encoding='utf-8')
    del data_9
    gc.collect()

    data_10 = data['2177']
    data_10.to_csv(r'D:/kaggle/health/col/2177.csv', sep=',', encoding='utf-8')
    del data_10
    gc.collect()

    data_11 = data['0546']
    data_11.to_csv(r'D:/kaggle/health/col/0546.csv', sep=',', encoding='utf-8')
    del data_11
    gc.collect()

    # data1 = pd.concat([data_1, data_2])
    # data1.to_csv(r'D:/kaggle/health/temp_data/vid_0102.csv', index=False, sep=',', encoding='utf-8')

    # part1 = pd.read_csv(r'D:/kaggle/health/temp_data/vid.csv', encoding='utf-8')
    # part2 = pd.read_csv(r'D:/kaggle/health/temp_data/0102.csv', encoding='utf-8')

    # merge = pd.concat([part1, part2])
    # del part1
    # del part2
    # gc.collect()
    # merge.to_csv(r'D:/kaggle/health/temp_data/vid_0102.csv', sep=',', encoding='utf-8')
    # print part2.shape (38198, 1)
    # print part1.shape (38198, 1)
