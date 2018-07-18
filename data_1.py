# coding:utf-8
# 选取数值特征，比较笨的方法。。。然后把对应的表头作为数值特征

import pandas as pd
import warnings
import sys
import gc

sys_new = reload(sys)
sys_new.setdefaultencoding('utf-8')

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    train = pd.read_csv(r'D:/kaggle/health/train_set_1.csv', encoding='utf-8')
    # test = pd.read_csv(r'D:/kaggle/health/test_set_1.csv', encoding='utf-8')
    # print train.info()
    # dtypes: float64(1038), object(1668)
    # print test.info()
    # dtypes: float64(1382), object(1324)
    temp = train.select_dtypes(float)
    temp['vid'] = train['vid']
    del train
    gc.collect()
    temp.to_csv(r'D:/kaggle/health/temp_data/train_other.csv', index=False, sep=',', encoding='utf-8')
    del temp
    gc.collect()
    # temp = test.select_dtypes(float)
    # temp['vid'] = test['vid']
    # del test
    # gc.collect()
    # temp.to_csv(r'D:/kaggle/health/temp_data/test_other.csv', index=False, sep=',', encoding='utf-8')
