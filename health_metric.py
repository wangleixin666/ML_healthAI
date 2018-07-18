# coding:utf-8

from math import log1p, pow
import pandas as pd
import warnings
import xgboost
# import lightgbm as lgb
import sys
import numpy as np

sys_new = reload(sys)
sys_new.setdefaultencoding('utf-8')

warnings.filterwarnings("ignore")


def calc_logloss(true_df, pred_df):
    loss_sum = 0
    rows = true_df.shape[0]
    for c in true_df.columns:
        # 预测结果必须要>0,否则log函数会报错，导致最终提交结果没有分数
        true_df[c] = true_df[c].apply(lambda x: log1p(x))
        pred_df[c] = pred_df[c].apply(lambda x: log1p(x))
        true_df[c + 'new'] = pred_df[c] - true_df[c]
        true_df[c + 'new'] = true_df[c + 'new'].apply(lambda x: pow(x, 2))
        loss_item = (true_df[c + 'new'].sum()) / rows
        loss_sum += loss_item
        # print('%s的loss：%f' % (c, loss_item))
    return loss_sum / 5


if __name__ == '__main__':
    train_data = pd.read_csv(r'D:/kaggle/health/train_set.csv')
    test_data = pd.read_csv(r'D:/kaggle/health/test_set.csv')

    features = ['Unnamed: 0', '004997', '100005', '300006', '300007', '315', '316', '317', '311', '319100', '33',
                '34', '39', '459155', '459156', '459158', '459159', '809008', '809009', '809010', '809013', '809017',
                '809018', '809019', '809020', '809022', '809023', '809026', '809027', '809029', '809031', '809032',
                '809033', '809034', '809035', '809037', '809038', '809039', '809040', '809041', '100007', '1124',
                '1125', '279006', '300070', '31', '310', '3184', '319', '37', '459154', '669001', '669004',
                '669005', '809042', '809043', '809044', '809045', '809046', '809047', '809048', '809049', '809050',
                '809051', '809053', '809054', '809055', '809056', '809057', '809058', '809059', '809060', '809061',
                '979010', '979024', '979025', '979026', '979027']
    labels = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']

    X_train = train_data[features]
    X_test = test_data[features]

    sum_loss = 0
    for k in range(0, 5, 1):
        # 交叉验证结果
        test_of_train_k = train_data[int(0.2 * k * train_data.index.max()):int(0.2 * (k + 1) * train_data.index.max())]
        # len_k = int(0.2*(k+1)*train_data.index.max()) - int(0.2*k*train_data.index.max())
        # print len_k
        # print int(0.2*k*train_data.index.max())
        # print int(0.2*(k+1)*train_data.index.max())

        # test_of_train.to_csv(r'D:/kaggle/health/test_test.csv', index=False, sep=',', encoding='utf-8')
        # print 'Done'
        # 生成测试集验证线下结果
        # test_test = pd.read_csv(r'D:/kaggle/health/test_test.csv')

        test_of_train_data_k = test_of_train_k[features]
        y_test_k = test_of_train_k[labels]
        # y_test_k.to_csv(r'D:/kaggle/health/temp_data/y_test_k.csv')
        # 还要对数据进行填充，把负值去除掉，而且只能一列一列做。。。
        # z这也太麻烦了，要每一个特征列都要填充
        # 其实只要对输入验证的结果进行非负替换就可以了啊。。

        for m in range(0, 5, 1):
            # 结果之后第5列需要处理，也就是0,1,2,3,4
            # 替换负值，还有要填充缺失值
            y_test_k.iloc[:, m].to_csv(r'D:/kaggle/health/temp_data/test2.csv')
            # y_test_k.iloc[:, m]取第m列所有行。。。终于
            for n in range(int(0.2 * k * train_data.index.max()), int(0.2 * (k + 1) * train_data.index.max()), 1):
                # 这个范围是取的范围。。
                y_test_k.iloc[:, m].fillna(np.round(y_test_k.iloc[:, m].mean(), 0), inplace=True)
                # 先填充，然后替换负值
                if y_test_k.iloc[:, m].get_value(n) <= 0 or y_test_k.iloc[:, m].get_value(n) > 250:
                    # 处理异常数据
                    # print y_test_k.iloc[:, m].get_value(n)
                    y_test_k.iloc[:, m].replace(y_test_k.iloc[:, m].get_value(n), np.round(y_test_k.iloc[:, m].mean()),
                                                inplace=True)
                    # 必须加上inplace=True否则本质上是不会变的
        y_test_k.to_csv(r'D:/kaggle/health/temp_data/y_test_k_1.csv')

        for i in range(0, 5, 1):
            y_train_column_i = train_data[labels[i]]
            if i == 0:
                y_train_column_i.replace(0, np.round(y_train_column_i.mean(), 0), inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)
                # 填补缺失值，0也是缺失值，容易造成误差
            if i == 1:
                for j in range(y_train_column_i.index.max()):
                    if y_train_column_i.get_value(j) > 150:
                        y_train_column_i.replace(y_train_column_i.get_value(j), 150, inplace=True)
                    if y_train_column_i.get_value(j) == 0:
                        y_train_column_i.replace(y_train_column_i.get_value(j), np.round(y_train_column_i.mean(), 0),
                                                 inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)

            # clf = xgboost.XGBRegressor()
            # clf = lgb.LGBMRegressor()
            clf = xgboost.XGBRegressor()

            clf.fit(X_train, y_train_column_i)

            test_of_train_k[labels[i]] = clf.predict(test_of_train_data_k)
            test_of_train_k[labels].to_csv(r'D:/kaggle/health/temp_data/y_pred_res_k_1.csv')
            # 只要对预测结果进行非负填充即可
            # 可能就是预测结果或者测试集的结果有负值，导致没法进行准确的评估
            # 可以采用我们提取出来的train_data和test_data，只有数值特征。。。
        y_test_k = pd.DataFrame(y_test_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])
        y_pred_res_k = pd.DataFrame(test_of_train_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])

        sum_loss += calc_logloss(y_test_k, y_pred_res_k)

        print sum_loss / 5

    """
    xgboost默认参数结果：
    舒张压的loss：0.023924
    收缩压的loss：0.019456
    血清甘油三酯的loss：0.112912
    血清高密度脂蛋白的loss：0.015392
    血清低密度脂蛋白的loss：0.042288
    ('averge loss: ', 0.071323950497048538)  
    0.0428978311947

    lightgbm默认参数，线上结果：
    0.0386509778327
    0.0374256840782
    0.0393193788227
    0.038909253976
    0.0389655968202
    xgboost默认参数，线上结果：
    0.0426437072114
    0.0427493291761
    0.0438698231955
    0.0440717876598
    0.0453902737265
    """
    # 是收缩压，完了才是舒张压