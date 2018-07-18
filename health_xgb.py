# coding:utf-8
# 对xgboost进行调参

from math import log1p, pow
import pandas as pd
import warnings
import xgboost
import sys
import numpy as np

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
    features = ['Unnamed: 0', '004997', '100005', '300006', '300007', '315', '316', '317', '311', '319100', '33',
                '34', '39', '459155', '459156', '459158', '459159', '809008', '809009', '809010', '809013', '809017',
                '809018', '809019', '809020', '809022', '809023', '809026', '809027', '809029', '809031', '809032',
                '809033', '809034', '809035', '809037', '809038', '809039', '809040', '809041', '100007', '1124',
                '1125', '279006', '300070', '31', '310', '3184', '319', '37', '459154', '669001', '669004',
                '669005', '809042', '809043', '809044', '809045', '809046', '809047', '809048', '809049', '809050',
                '809051', '809053', '809054', '809055', '809056', '809057', '809058', '809059', '809060', '809061',
                '979010', '979024', '979025', '979026', '979027']
    # 77列数值特征
    labels = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']
    train_data = train_all[0:30000]
    # 只有前30000行训练，看看结果
    X_train = train_data[features]
    sum_loss = 0

    for k in range(0, 5, 1):

        test_of_train_k = train_all[int(0.2*k*train_all.index.max()):int(0.2*(k+1)*train_all.index.max())]
        test_of_train_data_k = test_of_train_k[features]
        y_test_k = test_of_train_k[labels]

        for m in range(0, 5, 1):
            for n in range(int(0.2*k*train_all.index.max()), int(0.2*(k+1)*train_all.index.max()), 1):
                y_test_k.iloc[:, m].fillna(np.round(y_test_k.iloc[:, m].mean(), 0), inplace=True)
                if y_test_k.iloc[:, m].get_value(n) <= 0 or y_test_k.iloc[:, m].get_value(n) > 250:
                    y_test_k.iloc[:, m].replace(y_test_k.iloc[:, m].get_value(n), np.round(y_test_k.iloc[:, m].mean()),
                                                inplace=True)

        for i in range(0, 5, 1):
            y_train_column_i = train_data[labels[i]]
            if i == 0:
                y_train_column_i.replace(0, np.round(y_train_column_i.mean(), 0), inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)
                clf_i = xgboost.XGBRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 1:
                for j in range(y_train_column_i.index.max()):
                    if y_train_column_i.get_value(j) > 150:
                        y_train_column_i.replace(y_train_column_i.get_value(j), 150, inplace=True)
                    if y_train_column_i.get_value(j) == 0:
                        y_train_column_i.replace(y_train_column_i.get_value(j), np.round(y_train_column_i.mean(), 0),
                                                 inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)
                clf_i = xgboost.XGBRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 2:
                clf_i = xgboost.XGBRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 3:
                clf_i = xgboost.XGBRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)
            if i == 4:
                clf_i = xgboost.XGBRegressor()
                clf_i.fit(X_train, y_train_column_i)
                test_of_train_k[labels[i]] = clf_i.predict(test_of_train_data_k)

        y_test_k = pd.DataFrame(y_test_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])
        y_pred_res_k = pd.DataFrame(test_of_train_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])

        print calc_logloss(y_test_k, y_pred_res_k)
        # 第五个就相当于测试集了，前4个是训练集的结果验证，看看过拟合程度
    """
    xgboost默认参数，线下结果：0.0482772550949线上结果：0.04539    
    默认参数的结果：
    0.0421873846731
    0.042108182687
    0.0432793898885
    0.0438856653871
    线下测试集：0.0482772550949
    
    min_weight=2
    0.0421310240997
    0.0421385004594
    0.0433484555746
    0.0436977682826
    0.0484320516333
    
    =4:
    0.0421587594652
    0.0421628978794
    0.0433549607089
    0.0437283147458
    0.048301808698

    n_estimators=200
    0.0409507439422
    0.0409110217421
    0.0420116357318
    0.0425823720773
    0.0487550877722
    过拟合了
    
    learning_rate=0.05, max_depth=3, min_child_weight=4, 
    0.0432442319023
    0.0430530538676
    0.0444934662674
    0.0447160615669
    0.0482427967062
    
    只有leaning_rate=0.05
    0.0433274397541
    0.0430808339357
    0.04451417878
    0.0448381441137
    0.0483109438651
    
    只有learning_rate=0.15时
    0.0414523619896
    0.0414602393714
    0.0425325457301
    0.04317219343
    0.0486208972729
    
    max_depth=1
    0.0452023262344
    0.0447486601117
    0.0463748881405
    0.0464058463898
    0.0487015834425
    
    max_depth=5
    0.0391822866112
    0.0389044780697
    0.0402431174787
    0.0410905149437
    0.0486954401415
    weight=2
    0.0392500514393
    0.0389869158934
    0.040154938492
    0.0410343259673
    0.0486900560571
    """

    """
    对每一列单独选取模型，先考虑第三列的 血清甘油三酯 0.124746758202
    learning_rate为0.05 血清甘油三酯 0.124538814197 略好
    0.03 血清甘油三酯 0.123224566631   
    如果选用lightgbm则会有较好的效果
    血清甘油三酯 0.0987653472279
    血清甘油三酯 0.0966249742666
    血清甘油三酯 0.10317620163
    血清甘油三酯 0.103639533538
    血清甘油三酯 0.125848880554
    # 这过拟合也太严重了吧 
    """

    """
    默认参数的XGBOOST，考虑换个模型来拟合第三列，也就是i=2时
    收缩压 0.0192740776435
    舒张压 0.0223552012099
    血清甘油三酯 0.111823913936
    血清高密度脂蛋白 0.0152157613717
    血清低密度脂蛋白 0.0422679692042
    0.0421873846731
    收缩压 0.0200347015571
    舒张压 0.0224284046047
    血清甘油三酯 0.110895210755
    血清高密度脂蛋白 0.0149217199619
    血清低密度脂蛋白 0.0422608765561
    0.042108182687
    收缩压 0.0193153572788
    舒张压 0.0226476378241
    血清甘油三酯 0.114949378313
    血清高密度脂蛋白 0.0157524270176
    血清低密度脂蛋白 0.0437321490092
    0.0432793898885
    收缩压 0.0200851546133
    舒张压 0.0226629358436
    血清甘油三酯 0.11591418451
    血清高密度脂蛋白 0.0152223582412
    血清低密度脂蛋白 0.0455436937279
    0.0438856653871
    收缩压 0.0218996662111
    舒张压 0.0247978257712
    血清甘油三酯 0.124746758202
    血清高密度脂蛋白 0.019941309317
    血清低密度脂蛋白 0.0500007159734
    0.0482772550949
    """