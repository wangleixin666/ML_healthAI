# coding:utf-8
# 清洗数据中的坏数据
import pandas as pd
import gc
import numpy as np


def clean_label1(s):
    x = str(s)
    if '亚健康' in x:
        x = '2'
    if '健康' in x:
        x = '1'
    if '疾病' in x:
        x = '0'
    return x


def clean_label2(s):
    x = str(s)
    if '亚健康' in x:
        x = '2'
    if '健康' in x:
        x = '1'
    if '疾病' in x:
        x = '0'
    return x


# 数据清洗
def data_clean(df):
    """
    for c in ['2302']:
        # df[c] = str(df[c])
        df[c] = df[c].apply(clean_label1)
        df[c] = df[c].astype('float64')
    """
    for c in ['0101']:
        # df[c] = str(df[c])
        df[c] = df[c].apply(clean_label2)
        df[c] = df[c].astype('float64')
    return df

train_data = pd.read_csv(r'D:/kaggle/health/train_set_1.csv', sep=',')
train = data_clean(train_data)
del train_data
gc.collect()
train.to_csv(r'D:/kaggle/health/train_set_1.csv', index=False, sep=',', encoding='utf-8')
del train
gc.collect()

test_data = pd.read_csv(r'D:/kaggle/health/test_set_1.csv', sep=',')
test = data_clean(test_data)
del test_data
gc.collect()
test.to_csv(r'D:/kaggle/health/test_set_1.csv', index=False, sep=',', encoding='utf-8')
del test
gc.collect()
