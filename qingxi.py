# coding:utf-8
# 清洗数据中的坏数据
import pandas as pd
import gc
import numpy as np


def clean_label(s):
    x = str(s)
    if '+' in x:#16.04++
        i=x.index('+')
        x=x[0:i]
    if '>' in x:#> 11.00
        i=x.index('>')
        x=x[i+1:]
    if len(x.split('.'))>2:
        #2.2.8
        i=x.rindex('.')
        x=x[0:i]+x[i+1:]
    if ' ' in x:
        i = x.index(' ')
        x = x[0:i]
        if str(x).isdigit() is False:
            x = np.nan
    if str(x).isdigit() is False and len(str(x)) > 4:
        x = x[0:4]

    return x


# 数据清洗
def data_clean(df):
    for c in ['190']:
        # df[c] = str(df[c])
        df[c] = df[c].apply(clean_label)
        df[c] = df[c].astype('float64')
    return df


# data1 =pd.read_csv(r'D:/kaggle/health/wenben/1840.csv', sep=',')
# train = data_clean(data1)
# train.to_csv(r'D:/kaggle/health/wenben/1840.csv', index=False, sep=',', encoding='utf-8')
"""
train_data = pd.read_csv(r'D:/kaggle/health/train_set_2.csv', sep=',')
train = data_clean(train_data)
del train_data
gc.collect()
train.to_csv(r'D:/kaggle/health/train_set_3.csv', index=False, sep=',', encoding='utf-8')
del train
gc.collect()
"""
test_data = pd.read_csv(r'D:/kaggle/health/test_set_3.csv', sep=',')
test = data_clean(test_data)
del test_data
gc.collect()
test.to_csv(r'D:/kaggle/health/test_set_4.csv', index=False, sep=',', encoding='utf-8')
del test
gc.collect()
"""
'1814', '1815', '190', '191', '1840', '192', '1850', '1117', '193', '314', '1115', '183', '2174',
              '10002', '10003', '31', '312', '2333', '100006', '320', '1845', '320', '2406', '1345', '2420',
              '155', '360', '1127', '269013', '269011', '269006', '979012', '979021', '979016', '979004',
              '979007', '979020', '979017', '979008', '979011', '979009', '979003', '979015', '979023',
              '979022', '979001', '979014', '979013', '979018', '979019', '979002', '979006', '979005',
              '300021', '300019', '1106', '139', '1107', '300018', '143', '669002', '669001', '1474',
              '669006', '300017', '100014', '100012', '1112', '2386', '100013', '2177', '809025',
              '809021', '10009', '300001', '2409', '2376', '300013', '669021', '2302', '069023',
              '10014', '31', '339122', '339131', '699001', '709024', '819014', 'P19033'
"""