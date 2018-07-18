# coding:utf-8
# 对文本进行onehot编码
import pandas as pd
import gc
import jieba
# 采用结巴分词
from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer()


def chuli(df):
    for c in ['0101']:
        # df[c] = str(df[c])
        # df[c] = df[c].decode('GBK')
        df[c] = df[c].apply(fenci)

        # df[c] = ' '.join(df[c])
        # df[c] = df[c].encode('utf-8')
    return df


def fenci(s):
    s = str(s)
    # s = s.decode('gbk', errors='ignore')
    # 'gb1830'所含的比'gbk'要多，因此下面代码段采用了'gb1830'。
    # 'gbk' codec can't decode bytes in position 16-17: illegal multibyte sequence
    s = jieba.cut(s, cut_all=False)
    x = ' '.join(s)
    return x


def tfidf(df):
    for c in ['0101']:
        # df[c] = str(df[c])
        # df[c] = df[c].decode('GBK')
        df[c] = vector.fit_transform(df[c].values.astype('U'))

        # df[c] = ' '.join(df[c])
        # df[c] = df[c].encode('utf-8')
        # 可以用toarray()函数得到一个ndarray类型的完整矩阵。
    return df


train_data = pd.read_csv(r'D:/kaggle/health/wenben/0101.csv', sep=',')
# arr = enc.fit_transform(train_data[['0101']])
# 还是要每一行进行分词处理
data = chuli(train_data)
data.to_csv(r'D:/kaggle/health/wenben/0101_1.csv', index=False, sep=',', encoding='utf-8')
del train_data
gc.collect()
tfidf(data).to_csv(r'D:/kaggle/health/wenben/0101_2.csv', index=False, sep=',')
"""

train_data = pd.read_csv(r'D:/kaggle/health/train_set_1.csv', sep=',')
train = chuli(train_data)
del train_data
gc.collect()
tfidf(train).to_csv(r'D:/kaggle/health/train_set_1.csv', index=False, sep=',', encoding='utf-8')
del train
gc.collect()

test_data = pd.read_csv(r'D:/kaggle/health/test_set_1.csv', sep=',')
test = chuli(test_data)
del test_data
gc.collect()
tfidf(test).to_csv(r'D:/kaggle/health/test_set_1.csv', index=False, sep=',', encoding='utf-8')
del test
gc.collect()
"""