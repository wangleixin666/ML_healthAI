# coding:utf-8
# 对xgboost进行调参

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
    train_all = pd.read_csv(r'D:/kaggle/health/train_set_1.csv')
    features = ['Unnamed: 0', '004997', '100005', '100007', '1124', '1125', '279006',
                '300006', '300007', '300070', '31', '310', '311', '315', '316', '317', '3184',
                '319', '319100', '33', '34', '37', '39',
                '459154', '459155', '459156', '459158', '459159',
                '669001', '669004', '669005',
                '809008', '809009', '809010', '809013', '809017',
                '809018', '809019', '809020', '809022', '809023', '809026', '809027', '809029', '809031', '809032',
                '809033', '809034', '809035', '809037', '809038', '809039', '809040', '809041', '809042', '809043',
                '809044', '809045', '809046', '809047', '809048', '809049', '809050', '809051', '809053', '809054',
                '809055', '809056', '809057', '809058', '809059', '809060', '809061',
                '979010', '979024', '979025', '979026', '979027']
    # 81列数值特征
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
                clf_i = lgb.LGBMRegressor(learning_rate=0.15)
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 2:
                # 血清甘油三酯
                clf_i = lgb.LGBMRegressor(learning_rate=0.05)
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 3:
                # 血清高密度脂蛋白
                clf_i = lgb.LGBMRegressor(learning_rate=0.05)
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 4:
                # 血清低密度脂蛋白
                clf_i = lgb.LGBMRegressor(learning_rate=0.15)
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)

        y_test_k = pd.DataFrame(y_test_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])
        y_pred_res_k = pd.DataFrame(test_of_train_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])

        print calc_logloss(y_test_k, y_pred_res_k)
        # 第五个就相当于测试集了，前4个是训练集的结果验证，看看过拟合程度

    """
    # 34000这过拟合。。。。换成30000重新开始了
    默认参数 
    收缩压 0.0216743376086
    舒张压 0.0247091564771
    血清甘油三酯 0.12399863103
    血清高密度脂蛋白 0.0197655903713
    血清低密度脂蛋白 0.0494026119713
    0.0479100654916 # 线上0.04459
    学习率就定了
    0.1 收缩压 0.0216743376086
    0.15 舒张压 0.0247034540692
    0.05 血清甘油三酯 0.122793470508
    0.05 血清高密度脂蛋白 0.0197409870995
    0.15 血清低密度脂蛋白 0.0491975088077
    n_estimators=200 血清低密度脂蛋白 0.0491502915769
    150 血清高密度脂蛋白 0.019736790224
    # 30000的结果
    线下结果：0.0476116687974 # 线上结果0.04467 怎么还下降了呢。。
    # 换成31000试试结果
    不调参：0.0476397184951 
    调参后：0.0473110704812 # 不用n_estimators 0.0474113951616
    
    # 新的数据集 (其实也包含了清除后的数据，只不过可以用的数值列更多了)
    """

    """
    34000来训练的结果。。。。
    第一次调参之后(线上还不如没调参)：
    收缩压 0.0201750184281
    舒张压 0.0226739035444
    血清甘油三酯 0.114784146568
    血清高密度脂蛋白 0.0182566918888
    血清低密度脂蛋白 0.0448732834639
    # 这一列确实效果好了一点
    0.0441526087786
    
    默认的lightgbm，分别的结果：
    收缩压 0.0201750184281
    舒张压 0.0226860509268
    血清甘油三酯 0.114239301482
    血清高密度脂蛋白 0.0176594481484
    血清低密度脂蛋白 0.0457996677184
    0.0441118973408
    
    learning_rate=0.05, num_leaves=80 血清甘油三酯 0.11275725119
    100 血清高密度脂蛋白 0.016547981104
    150 血清低密度脂蛋白 0.0432836009591
    
    num_leaves=50   收缩压 0.0196506827439
    learning_rate=0.05, n_estimators=200 舒张压 0.0227126202937效果变差了，去掉循环200次
    learning_rate=0.05, num_leaves=100 血清甘油三酯 0.111430302669
    num_leaves=150 血清高密度脂蛋白 0.016154369277
    learning_rate=0.05, num_leaves=200 血清低密度脂蛋白 0.0423488092049
    
    100 收缩压 0.0188443246579
    0.05 5 depth=5不靠谱 舒张压 0.0240993288963
    0.05 150 血清甘油三酯 0.109536668496
    200 血清高密度脂蛋白 0.015849853323
    300 血清低密度脂蛋白 0.0412288576122
    
    500 收缩压 0.0167008387902
    0.05 500 血清甘油三酯 0.100773826205
    500 血清高密度脂蛋白 0.015013140469
    500 血清低密度脂蛋白 0.0397797517882
    
    # 感觉用34000训练还是不靠谱，过拟合太严重了，用30000训练能保证最后的8000行左右的数据确实是接近线上结果的
    收缩压 0.0232901533064
    舒张压 0.024755067606
    血清甘油三酯 0.128843472114
    血清高密度脂蛋白 0.0211903840773
    血清低密度脂蛋白 0.0501461882464
    0.04964505307
    """
