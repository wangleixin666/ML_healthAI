# coding:utf-8

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
        # 预测结果必须要>0,否则log函数会报错，导致最终提交结果没有分数
        true_df[c] = true_df[c].apply(lambda x: log1p(x))
        pred_df[c] = pred_df[c].apply(lambda x: log1p(x))
        true_df[c + 'new'] = pred_df[c] - true_df[c]
        true_df[c + 'new'] = true_df[c + 'new'].apply(lambda x: pow(x, 2))
        loss_item = (true_df[c + 'new'].sum()) / rows
        loss_sum += loss_item
        print('%s的loss：%f' % (c, loss_item))
    print('averge loss: ', loss_sum / 5)


if __name__ == '__main__':
    train_data = pd.read_csv(r'D:/kaggle/health/train_set.csv')
    test_data = pd.read_csv(r'D:/kaggle/health/test_set.csv')

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

    """
    把特征换成test_set中的特征，这样构建之间的关系可能结果更加准确
    因为test_set中的特征肯定在train_set中，并且有值让我们训练，
    不然就有可能出现结果为负的情况
    结果发现训练集中的特征，测试集里都有。。。而且测试集多了好多特征
    原来的：
    'Unnamed: 0', '004997', '100005', '300006', '300007',
    '315', '316', '317', '311', '319100', '33', '34', '39',
    '459155', '459156', '459158', '459159',
    '809008', '809009', '809010', '809013', '809017',
    '809018', '809019', '809020', '809022', '809023', '809026',
    '809027', '809029', '809031', '809032', '809033', '809034',
    '809035', '809037', '809038', '809039', '809040', '809041'
    增加了以后的：
    'Unnamed: 0', '004997', '100005', '300006', '300007', '315', '316', '317', '311', '319100', '33',
    '34', '39', '459155', '459156', '459158', '459159', '809008', '809009', '809010', '809013', '809017',
    '809018', '809019', '809020', '809022', '809023', '809026', '809027', '809029', '809031', '809032',
    '809033', '809034', '809035', '809037', '809038', '809039', '809040', '809041', '100007', '1124',
    '1125', '279006', '300070', '31', '310', '3184', '319', '37', '459154', '669001', '669004',
    '669005', '809042', '809043', '809044', '809045', '809046', '809047', '809048', '809049', '809050',
    '809051', '809053', '809054', '809055', '809056', '809057', '809058', '809059', '809060', '809061',
    '979010', '979024', '979025', '979026', '979027'
    test中的：
    004997	105	106	108	109	100005	100006	100007	100013	1110	1112	1124	1125	1844	
    20002	2165	2390	269004	269005	269006	269007	269008	269009	269010	269012	269014	
    269015	269016	269017	269018	269019	269020	269021	269022	269023	269024	269025	279006	
    2986	300006	300007	300008	300009	300011	300012	300070	300074	300076	300092	
    31	310	311	313	315	316	317	3184	
    319	319100	32	320	33	34	37	38	39	459154	459155	459156	459158	459159	669001	669004	
    669005	669007	669008	669009	669021	809002	809003	809008	809009	809010	809013	809017	
    809018	809019	809020	809022	809023	809024	809026	809027	809029	809031	809032	809033	
    809034	809035	809037	809038	809039	809040	809041	809042	809043	809044	809045	809046	
    809047	809048	809049	809050	809051	809052	809053	809054	809055	809056	809057	809058	
    809059	809060	809061	979010	979024	979025	979026	979027
    """

    X_train = train_data[features]
    X_test = test_data[features]
    labels = ['舒张压', '收缩压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']

    test_of_train = train_data[0:8000]
    # test_of_train.to_csv(r'D:/kaggle/health/test_test.csv', index=False, sep=',', encoding='utf-8')
    # print 'Done'
    # 生成测试集验证线下结果

    test_test = pd.read_csv(r'D:/kaggle/health/temp_data/test_test.csv')
    test_of_train_data = test_of_train[features]
    y_test = test_of_train[labels]

    for i in range(0, 5, 1):
        y_train_column_i = train_data[labels[i]]
        if i == 0:
            y_train_column_i.replace(0, np.round(y_train_column_i.mean(), 0), inplace=True)
            y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)
            # 填补缺失值，如果不填补训练集中的缺失值，会造成结果为0的情况。。。。
            # 0也是缺失值，容易造成误差
        if i == 1:
            for j in range(y_train_column_i.index.max()):
                # 观察到有异常数据，对异常数据进行处理，替换成均值，会有一定的误差
                if y_train_column_i.get_value(j) > 150:
                    y_train_column_i.replace(y_train_column_i.get_value(j), 150, inplace=True)
                if y_train_column_i.get_value(j) == 0:
                    y_train_column_i.replace(y_train_column_i.get_value(j), np.round(y_train_column_i.mean(), 0),
                                             inplace=True)
            y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)

        clf = lgb.LGBMRegressor()

        clf.fit(X_train, y_train_column_i)

        test_test[labels[i]] = clf.predict(test_of_train_data)

    y_test = pd.DataFrame(y_test, columns=['舒张压', '收缩压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])
    y_pred_res = pd.DataFrame(test_test, columns=['舒张压', '收缩压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])
    calc_logloss(y_test, y_pred_res)

    """
    舒张压的loss：0.021483
    收缩压的loss：0.017924
    血清甘油三酯的loss：0.101209
    血清高密度脂蛋白的loss：0.014118
    血清低密度脂蛋白的loss：0.038208
    ('averge loss: ', 0.064314139673737114)

    血清甘油三酯的结果每次都比较差
    # 可以尝试对它单独调参
    """

    """
    n_estimators RF最大的决策树个数
    n_estimators太小，容易欠拟合，n_estimators太大，计算量会太大
    并且n_estimators到一定的数量后，再增大n_estimators获得的模型提升会很小，所以一般选择一个适中的数值,默认是100
    mx_depth是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本，一般不用考虑
    num_leaves = 2^(max_depth),就可以完成depth-wise tree growth和leaf-wise growth的转换
    但是在实践中这种简单的转换并不能够有好的结果，设置相同的时候，leaf-wise growth树会比depth-wise tree growth要深很多，容易过拟合
    所以在调这个参数的时候，策略就是num_leaves < 2^(max_depth)，一般比100大
    较大的max_bin(但会是模型速度下降)能提高精度
    使用较小max_bin.能避免过拟合
    使用较小 num_leaves 能避免过拟合
    min_data_in_leaf设置的大则可以避免建立过于深的树，但是会造成欠拟合，一般设置100以上比较合适。
    """

    """
    舒张压的loss：0.021483
    收缩压的loss：0.017924
    血清甘油三酯的loss：0.101209
    血清高密度脂蛋白的loss：0.014118
    血清低密度脂蛋白的loss：0.038208
    ('averge loss: ', 0.03858848380424227)
    线上0.0446
    """
