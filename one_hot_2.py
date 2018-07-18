# coding:utf-8
# 对文本进行onehot编码
import pandas as pd
import gc
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv(r'D:/kaggle/health/wenben/0101_2.csv', sep=',')
# print train_data.shape
# (9538, 2)
# print train_data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9538 entries, 0 to 9537
Data columns (total 2 columns):
Unnamed: 0    9538 non-null int64
0101          9538 non-null object
dtypes: int64(1), object(1)
memory usage: 149.1+ KB
None
"""
# print train_data.head()
"""
Unnamed: 0                                               0101
0                               0    (0, 2199)\t0.034306623084\r\n  (0, 1681)\t0....
1                               1    (0, 2199)\t0.034306623084\r\n  (0, 1681)\t0....
2                               2    (0, 2199)\t0.034306623084\r\n  (0, 1681)\t0....
3                               3    (0, 2199)\t0.034306623084\r\n  (0, 1681)\t0....
4                               4    (0, 2199)\t0.034306623084\r\n  (0, 1681)\t0....
"""
# 然后我们把第二行转换成列就是我们要的特征了吧。。。。
data = train_data.iloc[0, :]
# Name: 0101, Length: 9538, dtype: object
"""
Unnamed: 0                                                    1
0101            (0, 2199)\t0.034306623084\r\n  (0, 1681)\t0....
Name: 1, dtype: object
"""
# print data.shape
# (2L,)
