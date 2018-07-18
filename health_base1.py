#!/usr/bin/env python    # -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import sys
sys_new = reload(sys)
sys_new.setdefaultencoding('utf-8')

# 读取数据
train = pd.read_csv(r'D:/kaggle/health/meinian_round1_train_20180408.csv', sep=',', encoding='gbk')
test = pd.read_csv(r'D:/kaggle/health/meinian_round1_test_a_20180409.csv', sep=',', encoding='gbk')

merge_part1_2 = pd.read_csv(r'D:/kaggle/health/merge_part1_2.csv')

# 找到train，test各自属性进行拼接
train_of_part = merge_part1_2[merge_part1_2['vid'].isin(train['vid'])]
test_of_part = merge_part1_2[merge_part1_2['vid'].isin(test['vid'])]

train = pd.merge(train, train_of_part, on='vid')
test = pd.merge(test, test_of_part, on='vid')


# 清洗训练集中的五个指标
def clean_label(x):
    x = str(x)
    if '+' in x:
        # 16.04++
        i = x.index('+')
        x = x[0:i]
    if '>' in x:
        # > 11.00
        i = x.index('>')
        x = x[i+1:]
    if len(x.split('.')) > 2:
        # 2.2.8
        i = x.rindex('.')
        x = x[0:i]+x[i+1:]
    if u'未做' in x or u'未查' in x or u'弃查' in x:
        x = np.nan
    if str(x).isdigit() is False and len(str(x)) > 4:
        x = x[0:4]
    return x


# 数据清洗
def data_clean(df):
    for c in [u'收缩压', u'舒张压', u'血清甘油三酯', u'血清高密度脂蛋白', u'血清低密度脂蛋白']:
        df[c] = df[c].apply(clean_label)
        df[c] = df[c].astype('float64')
    return df

train = data_clean(train)

print('---------------保存train_set和test_set---------------------')
train.to_csv(r'D:/kaggle/health/train_set.csv', index=False, encoding='utf-8')
test.to_csv(r'D:/kaggle/health/test_set.csv', index=False, encoding='utf-8')
