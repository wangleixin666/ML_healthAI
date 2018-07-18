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
    temp = pd.read_csv(r'D:/kaggle/health/temp_data/tmp.csv')
    # float64(88), int64(1)
    # 应该是83列，少了7列
    # print train.info()
    # float64(134), int64(1)
    train_set = temp.select_dtypes(float)
    del temp
    gc.collect()
    train_set['vid'] = temp['vid']
    train_set[['vid', ]].to_csv(r'D:/kaggle/health/temp_data/train_data_3.csv', index=False, sep=',', encoding='utf-8')
    del train_set
    gc.collect()

    """
    train_set = train.select_dtypes(float)
    train_set['vid'] = train['vid']
    train_set.to_csv(r'D:/kaggle/health/train_data.csv', encoding='utf-8')

    test = pd.read_csv(r'D:/kaggle/health/test_set.csv')
    test_set = test.select_dtypes(float)
    test_set['vid'] = test['vid']
    test_set.to_csv(r'D:/kaggle/health/test_data.csv', encoding='utf-8')
    """
