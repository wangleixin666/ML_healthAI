# coding:utf-8
# 对xgboost进行调参

from math import log1p, pow
import pandas as pd
import warnings
import xgboost
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

    features = ['2403', '2404', '2405', '1814', '1815', '190', '191', '1840', '192', '1850', '10004',
                '1117', '193', '314', '1115', '183', '2174', '10002', '10003', '31', '315',
                '316', '319', '38', '312', '32', '2333', '100006', '313', '320', '1845',
                '2372', '1845', '320', '100007', '2406', '100005', '37', '317', '33', '34',
                '39', '2302']
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
            if i == 1:
                # 舒张压
                for j in range(y_train_column_i.index.max()):
                    if y_train_column_i.get_value(j) > 150:
                        y_train_column_i.replace(y_train_column_i.get_value(j), 150, inplace=True)
                    if y_train_column_i.get_value(j) == 0:
                        y_train_column_i.replace(y_train_column_i.get_value(j), np.round(y_train_column_i.mean(), 0),
                                                 inplace=True)
                y_train_column_i.fillna(np.round(y_train_column_i.mean(), 0), inplace=True)
            clf = lgb.LGBMRegressor()
            clf.fit(X_train, y_train_column_i)
            # imp = pd.DataFrame()
            # imp['s'] = list(clf.feature_importances_)
            # print(imp.sort_values('s', ascending=False))
            test_of_train_k[labels[i]] = clf.predict(test_of_train_data_k)

        y_test_k = pd.DataFrame(y_test_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])
        y_pred_res_k = pd.DataFrame(test_of_train_k, columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'])

        print calc_logloss(y_test_k, y_pred_res_k)

"""
三列数值特征的结果：

收缩压 0.0191823539534
舒张压 0.022377287221
血清甘油三酯 0.11141685878
血清高密度脂蛋白 0.0187401656251
血清低密度脂蛋白 0.0506450977088
0.0444723526577
KeyError: "['1840'] not in index"

7列
收缩压 0.0190738663682
舒张压 0.0224238679286
血清甘油三酯 0.11078813647
血清高密度脂蛋白 0.0186861975589
血清低密度脂蛋白 0.0505975814374
0.0443139299526

11列
收缩压 0.0190842500245
舒张压 0.0224222425687
血清甘油三酯 0.110758157285
血清高密度脂蛋白 0.0186993819113
血清低密度脂蛋白 0.0506099998812
0.044314806334

到319
收缩压 0.0189602305976
舒张压 0.0220988429707
血清甘油三酯 0.108108500515
血清高密度脂蛋白 0.0183725218827
血清低密度脂蛋白 0.050100199772
0.0435280591477

到39
收缩压 0.0189145137517
舒张压 0.0220510071303
血清甘油三酯 0.107209292553
血清高密度脂蛋白 0.0180868692108
血清低密度脂蛋白 0.0489425646673
0.0430408494626

'300007', '1345', '2420', '37', '317', '33',
'34', '2177', '809021', '300001', '2376'
后面加的数值列感觉没什么效果了，反而更差了

加了好多咧之后
'269020', '269018', '269004', '269023', '269016', '269005', '269007'
'269024', '269022', '269021', '269015', '269025', '269013', '269011', '269003',
'269017', '269006', '269012', '269008', '269009', '269010', '269019', '269014', 
这些列居然都没效果。。。

收缩压 0.0189238468362
舒张压 0.0220680764062
血清甘油三酯 0.107911355411
血清高密度脂蛋白 0.0181586089658
血清低密度脂蛋白 0.0486803481402
0.0431484471518
2   265
19  256
38  254
39  232
20  215
0   200
4   199
37  189
22  185
41  178
1   175
21  168
40  167
42  131
43   71
3    53
14   20
11   13
6    10
13    9
44    6
8     4
'2403', '2404', '2405', '1814', '1815', '191', '192', '1117', '314', '1115', '31', '315', 
'316', '319', '39', '300007', '1345', '2420', '37', '317', '33', '34', 
# 72列特征就只有这些列起作用。。。22列
'2403', '2404', '2405', '1814', '1815', '190', '191', '1840', '192', '1850', '10004',
'1117', '193', '314', '1115', '183', '2174', '10002', '10003', '31', '315', 
'316', '319', '38', '312', '32', '2333', '100006', '313', '320', '1845', 
'2372', '1845', '320', '100007', '2406', '100005', '37', '317', '33', '34', 
'39', '300007', '1345', '2420', '155', '360', '1127', '269020', '269018', '269004', 
'269023', '269016', '269005', '269007', '269024', '269022', '269021', '269015', '269025', '269013', 
'269011', '269003', '269017', '269006', '269012', '269008', '269009', '269010', '269019', '269014'

又加了好多列之后：
收缩压 0.0189321605791
舒张压 0.0220899600679
血清甘油三酯 0.108144642751
血清高密度脂蛋白 0.0181393012521
血清低密度脂蛋白 0.0487897720725
0.0432191673445
'979012', '979021', '979016', '979004', '979007', '979020', '979017', '979008', '979011',
'979009', '979003', '979015', '979023', '979022', '979001', '979014', '979013', '979018',
'979019', '979002', '979006', '979005', '300021', '300019', '1106', '269014', '269016',
'139', '809001', '1107', '300018', '143', '669002', '669001', '1474', '669006', '300017',
'100014', '809009', '809008', '100012', '1112', '669005', '809004', '2386', '100013',
'2177', '809017', '669009', '300092', '809025', '809026', '809023', '809021', '10009',
'300008', '300001', '2409', '2376', '300013', '300012', '300011', '669004', '669021', '809010'

'2403', '2404', '2405', '1814', '1815', '191', '192', '1117', '314', '1115', '31',
'315', '316', '319', '39', 
如果只这样，效果一般0.0435984768011，应该是和有的特征咧不够独立
"""