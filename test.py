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
            x=np.nan
    # if '未做' in x or '未查' in x or '弃查' in x:
    #    x = np.nan
    if str(x).isdigit() is False and len(str(x)) > 4:
        x = x[0:4]
    return x


# 数据清洗
def data_clean(df):
    for c in ['2403', '2404', '2405', '1814', '1815', '190', '191', '1840', '192', '1850', '10004',
              '1117', '193', '314', '1115', '183', '2174', '10002', '10003', '31', '315',
              '316', '319', '38', '312', '32', '2333', '100006', '313', '320', '1845',
              '2372', '1845', '320', '100007', '2406', '100005', '37', '317', '33', '34',
              '39', '300007', '1345', '2420', '155', '360', '1127', '269020', '269018', '269004',
              '269023', '269016', '269005', '269007', '269024', '269022', '269021', '269015', '269025', '269013',
              '269011', '269003', '269017', '269006', '269012', '269008', '269009', '269010', '269019', '269014',
              '979012', '979021', '979016', '979004', '979007', '979020', '979017', '979008', '979011',
              '979009', '979003', '979015', '979023', '979022', '979001', '979014', '979013', '979018',
              '979019', '979002', '979006', '979005', '300021', '300019', '1106', '269014', '269016',
              '139', '809001', '1107', '300018', '143', '669002', '669001', '1474', '669006', '300017',
              '100014', '809009', '809008', '100012', '1112', '669005', '809004', '2386', '100013',
              '2177', '809017', '669009', '300092', '809025', '809026', '809023', '809021', '10009',
              '300008', '300001', '2409', '2376', '300013', '300012', '300011', '669004', '669021', '809010', '2302']:
        # df[c] = str(df[c])
        df[c] = df[c].apply(clean_label)
        df[c] = df[c].astype('float64')
    return df


# data1 =pd.read_csv(r'D:/kaggle/health/wenben/1840.csv', sep=',')
# train = data_clean(data1)
# train.to_csv(r'D:/kaggle/health/wenben/1840.csv', index=False, sep=',', encoding='utf-8')

train_data = pd.read_csv(r'D:/kaggle/health/train_set_1.csv', sep=',')
train = data_clean(train_data)
del train_data
gc.collect()
train.to_csv(r'D:/kaggle/health/train_set_5.csv', index=False, sep=',', encoding='utf-8')
del train
gc.collect()
"""
test_data = pd.read_csv(r'D:/kaggle/health/test_set_2.csv', sep=',')
test = data_clean(test_data)
del test_data
gc.collect()
test.to_csv(r'D:/kaggle/health/test_set_3.csv', index=False, sep=',', encoding='utf-8')
del test
gc.collect()
"""

"""
用这种规则清洗-1的结果变成-5，然后与-4进行对比
"""