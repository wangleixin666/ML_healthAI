# coding:utf-8
# 对lightgbm进行调参

from math import log1p, pow
import pandas as pd
import warnings
import lightgbm as lgb
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
                y_train_column_i.replace(0, np.round(y_train_column_i.mean(), 0), inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)
            if i == 1:
                for j in range(y_train_column_i.index.max()):
                    if y_train_column_i.get_value(j) > 150:
                        y_train_column_i.replace(y_train_column_i.get_value(j), 150, inplace=True)
                    if y_train_column_i.get_value(j) == 0:
                        y_train_column_i.replace(y_train_column_i.get_value(j), np.round(y_train_column_i.mean(), 0),
                                                 inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)

            clf_1 = lgb.LGBMRegressor()
            clf_1.fit(X_train, y_train_column_i)
            # 用前30000行训练模型
            test_of_train_k[labels[i]] = clf_1.predict(test_of_train_data_k)
            # 然后来拟合全部的38000行左右的数据，第五部分就相当于测试集验证拟合程度了

            # 观察到每次的i=0,1,3结果较好，第三列，也就是i=2时，结果比较差，i=4时结果一般，，考虑单独调参

        y_test_k = pd.DataFrame(y_test_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])
        y_pred_res_k = pd.DataFrame(test_of_train_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])

        print calc_logloss(y_test_k, y_pred_res_k)
        # 第五个就相当于测试集了，前4个是训练集的结果验证，看看过拟合程度

    """
    lightgbm默认参数，线下总的平均结果（全部数据进行训练）：0.0386191539237
    默认参数的结果：
    0.0374119882602
    0.0361436934466
    0.0378249604451
    0.0376548049495
    线下测试集：0.0479100654916//然后线上结果为0.04459
    # 调参之后：0.0477050032188 线上结果0.04444
    learning_rate =0.05
    0.0394647896042
    0.038698661274
    0.0404450914752
    0.0401860090568
    0.0477761466728，比0.1的时候要好一点点,0.15要差很多
    num_leaves=50
    0.0377544776398
    0.0365021899532
    0.0384988825642
    0.0383152836198
    0.047678219536
    80
    0.0356223121609
    0.0338760333228
    0.0362790098131
    0.0361327112214
    0.0477260717051    
    n_estimators=200
    0.0348374730646
    0.0331262522868
    0.0353143211609
    0.0353142456184
    0.0479685770255 
    subsample=0.9,
    colsample_bytree=0.9
    0.0480449013522
    reg_alpha=2
    0.047961488448
    10
    0.0478820193667
    20
    0.047875191053
    加上reg_lambda=10
    0.0478660152737
    50
    0.0477436917003
    max_depth=7
    0.0477050032188
    
    learning_rate=0.05,
    num_leaves=50,
    n_estimators=200,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=20,
    reg_lambda=50,
    max_depth=7
    """
