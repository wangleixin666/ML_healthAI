# coding:utf-8
# 测试一波多少的训练集能做到同增同减

from math import log1p, pow
import pandas as pd
import warnings
import xgboost
# from sklearn.ensemble import GradientBoostingRegressor
# 还要填补缺失值
import sys
import numpy as np
import lightgbm as lgb

sys_new = reload(sys)
sys_new.setdefaultencoding('utf-8')
warnings.filterwarnings("ignore")


def calc_logloss(true_df, pred_df):
    loss_sum = 0
    rows = true_df.shape[0]
    for c in true_df.columns:
        true_df[c] = true_df[c].apply(lambda x: log1p(x))
        pred_df[c] = pred_df[c].apply(lambda x: log1p(x))
        true_df[c + 'new'] = pred_df[c] - true_df[c]
        true_df[c + 'new'] = true_df[c + 'new'].apply(lambda x: pow(x, 2))
        loss_item = (true_df[c + 'new'].sum()) / rows
        loss_sum += loss_item
        print c, loss_item
    return loss_sum / 5


if __name__ == '__main__':
    train_all = pd.read_csv(r'D:/kaggle/health/train_set.csv')
    features = ['Unnamed: 0', '004997', '100005', '100007', '1124', '1125', '279006',
                '300006', '300007', '300070', '31', '310', '311', '315', '316', '317', '3184',
                '319', '319100', '33', '34', '37', '39',
                '459154', '459155', '459156', '459158', '459159',
                '669001', '669004', '669005',
                '809004', '809007', '809008', '809009', '809010', '809013', '809017',
                '809018', '809019', '809020', '809022', '809023', '809026', '809027', '809029', '809031', '809032',
                '809033', '809034', '809035', '809037', '809038', '809039', '809040', '809041', '809042', '809043',
                '809044', '809045', '809046', '809047', '809048', '809049', '809050', '809051', '809053', '809054',
                '809055', '809056', '809057', '809058', '809059', '809060', '809061',
                '979010', '979024', '979025', '979026', '979027']
    # 78列数值特征 # 总共89列 # 81列，还有5列labels
    labels = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']
    train_data = train_all[0:31000]
    # 只有前30000行训练，看看结果
    X_train = train_data[features]
    sum_loss = 0

    for k in range(0, 5, 1):

        test_of_train_k = train_all[int(0.2 * k * train_all.index.max()):int(0.2 * (k + 1) * train_all.index.max())]
        test_of_train_data_k = test_of_train_k[features]
        y_test_k = test_of_train_k[labels]

        for m in range(0, 5, 1):
            for n in range(int(0.2 * k * train_all.index.max()), int(0.2 * (k + 1) * train_all.index.max()), 1):
                y_test_k.iloc[:, m].fillna(np.round(y_test_k.iloc[:, m].mean(), 0), inplace=True)
                if y_test_k.iloc[:, m].get_value(n) <= 0 or y_test_k.iloc[:, m].get_value(n) > 250:
                    y_test_k.iloc[:, m].replace(y_test_k.iloc[:, m].get_value(n), np.round(y_test_k.iloc[:, m].mean()),
                                                inplace=True)

        for i in range(0, 5, 1):
            y_train_column_i = train_data[labels[i]]
            if i == 0:
                # 收缩压
                y_train_column_i.replace(0, np.round(y_train_column_i.mean(), 0), inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)
                clf_i = lgb.LGBMRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 1:
                # 舒张压
                for j in range(y_train_column_i.index.max()):
                    if y_train_column_i.get_value(j) > 150:
                        y_train_column_i.replace(y_train_column_i.get_value(j), 150, inplace=True)
                    if y_train_column_i.get_value(j) == 0:
                        y_train_column_i.replace(y_train_column_i.get_value(j), np.round(y_train_column_i.mean(), 0),
                                                 inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)
                clf_i = lgb.LGBMRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 2:
                # 血清甘油三酯
                clf_i = lgb.LGBMRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 3:
                # 血清高密度脂蛋白
                clf_i = lgb.LGBMRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 4:
                # 血清低密度脂蛋白
                clf_i = lgb.LGBMRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)

        y_test_k = pd.DataFrame(y_test_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])
        y_pred_res_k = pd.DataFrame(test_of_train_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])

        print calc_logloss(y_test_k, y_pred_res_k)
