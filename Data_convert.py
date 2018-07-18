# coding:utf-8

import pandas as pd
from tqdm import tqdm
import time
import warnings

"""今天考虑对模型进行改进，对模型融合，LR + Xgboost"""
warnings.filterwarnings("ignore")


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])
    return data


def transform(self, df):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        feature_tuple = []
        for cat in self.categorical:
            if pd.notnull(row[cat]):
                feature = '{}_{}'.format(cat, row[cat])
                feature_tuple.append((self.field_index[cat], self.feature_index[feature], 1))
        for num in self.numerical:
            if pd.notnull(row[num]):
                feature_tuple.append((self.field_index[num], self.feature_index[num], row[num]))

        X.append(feature_tuple)
        y.append(row[self.target])
    return X, y


if __name__ == '__main__':

    data = pd.read_csv('al_train.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    train = data.loc[data.day < 24]  # 18,19,20,21,22,23 # 一共420693行，32列
    test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集 # 一共420693

    features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                ]
    target = ['is_trade']

    X_train = train[features]
    X_test = test[features]
    Y_train = train[target]
    Y_test = test[target]

    ffm_data = transform(X_train, Y_train)
    print ffm_data
