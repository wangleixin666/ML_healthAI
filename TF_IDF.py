from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

vector = TfidfVectorizer()


def tfidf(df):
    for c in ['0101']:
        # df[c] = str(df[c])
        # df[c] = df[c].decode('GBK')
        df[c] = vector.fit_transform(df[c].values.astype('U'))

        # df[c] = ' '.join(df[c])
        # df[c] = df[c].encode('utf-8')
    return df


train_data = pd.read_csv(r'D:/kaggle/health/wenben/0101_1.csv', sep=',', encoding='utf-8')
tfidf(train_data).to_csv(r'D:/kaggle/health/wenben/0101_2.csv', index=False, sep=',')
