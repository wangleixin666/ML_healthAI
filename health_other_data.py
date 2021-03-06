# coding:utf-8
# 对xgboost进行调参

from math import log1p, pow
import pandas as pd
import warnings
import xgboost
import sys
import numpy as np
from sklearn.decomposition import PCA
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
    train_all = pd.read_csv(r'D:/kaggle/health/train_set_5.csv')
    features = ['004997', '2403', '2404', '2405', '1814', '1815', '190', '191', '1840', '192', '1850', '10004',
                '1117', '193', '314', '1115', '183', '2174', '10002', '10003', '31', '315',
                '316', '319', '38', '312', '32', '2333', '100006', '313', '320', '1845',
                '2372', '1845', '320', '100007', '2406', '100005', '37', '317', '33', '34',
                '39', '300007', '1345', '2420', '155', '360', '1127', '269020', '269018', '269004',
                '269023', '269016', '269005', '269007', '269024', '269022', '269021', '269015', '269025', '269013',
                '269011', '269003', '269017', '269006', '269012', '269008', '269009', '269010', '269019', '269014',
                '979012', '979021', '979016', '979004', '979007', '979020', '979017', '979008', '979011',
                '979009', '979003', '979015', '979023', '979022', '979001', '979014', '979013', '979018',
                '979019', '979002', '979006', '979005', '300021', '300019', '1106', '269014', '269016',
                '139', '809001', '1107', '300018', '143', '669002', '669001', '1474', '669006', '300017',
                '100014', '809009', '809008', '100012', '1112', '669005', '809004', '2386', '100013',
                '2177', '809017', '669009', '300092', '809025', '809026', '809023', '809021', '10009',
                '300008', '300001', '2409', '2376', '300013', '300012', '300011', '669004', '669021', '809010', '2302',
                '0105', '0106', '0107', '0108', '0109', '0111', '0112', '019001', '019002', '019003', '019004',
                '019007',
                '019008', '019017', '019032', '019033', '019034', '019035', '019036', '019037', '019038', '019039',
                '019040', '019041',
                '019042', '019043', '019044', '019045', '019046', '019047', '019048', '019049', '019050', '019051',
                '019052', '019053',
                '019054', '019055', '019056', '019059', '019062', '069002', '069003', '069004', '069005', '069007',
                '069008', '069010',
                '069023', '069044', '069049', '069050', '1', '10000', '100005', '100007', '10005', '10012', '10014',
                '1123', '1124',
                '1125', '1136', '129056', '129057', '129058', '129079', '131', '134', '1343', '1346', '1349', '1359',
                '137', '1456',
                '1461', '1471', '159053', '159063', '179176', '179177', '179178', '179226', '1816', '184', '1849',
                '186', '1915',
                '1918', '199118', '20000', '21A002', '21A012', '21A021', '229080', '2392', '2451', '2452', '2453',
                '2454', '269028',
                '269029', '269030', '269031', '269052', '269055', '269056', '269057', '269058', '279001', '279002',
                '279003', '279004',
                '279005', '279006', '279028', '299168', '300003', '300006', '300007', '300015', '300022', '300037',
                '300038', '300039',
                '300069', '300070', '300072', '300075', '300077', '300080', '300087', '300111', '300114', '300117',
                '300118', '300119',
                '300129', '300136', '300146', '300166', '300168', '300169', '300170', '300171', '300172', '300174',
                '300175', '300176',
                '300178', '300179', '300180', '300181', '300182', '300183', '300184', '300185', '300186', '300187',
                '300188', '300189',
                '300190', '31', '310', '311', '315', '316', '317', '3184', '319', '319100', '319159', '319273', '3205',
                '3206', '3211',
                '3217', '33', '3302', '339105', '339106', '339107', '339114', '339122', '339125', '339126', '339128',
                '339129', '339130',
                '339131', '339135', '34', '346', '35', '36', '369007', '369008', '369085', '369098', '369108', '37',
                '378', '3814',
                '3816', '3818', '39', '419008', '439011', '439015', '439016', '439035', '459116', '459117', '459141',
                '459154', '459155',
                '459156', '459158', '459159', '459161', '459181', '459182', '459183', '459184', '459206', '459207',
                '459208', '459209',
                '459211', '459327', '459329', '459330', '459331', '459332', '459333', '459336', '459337', '459338',
                '459340', '459342',
                '509006', '509013', '539004', '559007', '559046', '559047', '669001', '669004', '669005', '669010',
                '669014', '669043',
                '669044', '669045', '669046', '699001', '699003', '699004', '699005', '699006', '699009', '709004',
                '709013', '709016',
                '709019', '709020', '709022', '709023', '709024', '709025', '709027', '709030', '709031', '709043',
                '709044', '709048',
                '729028', '739005', '759001', '769008', '769019', '809004', '809005', '809006', '809007', '809008',
                '809009', '809010',
                '809011', '809012', '809013', '809014', '809015', '809016', '809017', '809018', '809019', '809020',
                '809022', '809023',
                '809026', '809027', '809028', '809029', '809030', '809031', '809032', '809033', '809034', '809035',
                '809036', '809037',
                '809038', '809039', '809040', '809041', '809042', '809043', '809044', '809045', '809046', '809047',
                '809048', '809049',
                '809050', '809051', '809053', '809054', '809055', '809056', '809057', '809058', '809059', '809060',
                '809061', '819006',
                '819008', '819009', '819010', '819011', '819012', '819013', '819014', '819015', '819016', '819017',
                '819019', '819020',
                '819021', '819022', '819023', '819024', '819025', '819026', '819027', '819028', '819029', '819030',
                '819031', '839018',
                '899021', '899022', '909001', '979010', '979024', '979025', '979026', '979027', '979029', '979091',
                '979092', '979093',
                '979094', '979095', '979096', '989001', '989002', '989003', '989043', '989065', 'A49018', 'C19103',
                'C39002', 'D29008',
                'D29009', 'D29010', 'D29011', 'G49050', 'I19027', 'I69003', 'I69004', 'I69005', 'J29018', 'K29002',
                'L19008', 'P19033',
                'P79002', 'Q99001', 'Q99002', 'T99001', 'T99002', 'U99009', 'X19001', 'X19002', 'X19003', 'X19011',
                'Y79001',
                ]
    # 不调参
    # 81列数值特征  ：0.0476397184951
    # 169列数值特征 ：0.0472311235482
    # 400+列 ：0.0472163369872      0.04363
    # 增加了三个特征：0.0423527167071 0.0392

    labels = ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']
    train_data = train_all[0:31000]
    # 只有前30000行训练，看看结果
    X_train = train_data[features]

    # estimator = PCA(n_components=200)
    # 初始化一个可以将高维度特征向量压缩为低维度的PCA，设置为2维
    # X_train = estimator.fit_transform(X_train)
    # Input contains NaN, infinity or a value too large for dtype('float64').

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
        # 第五个就相当于测试集了，前4个是训练集的结果验证，看看过拟合程度
        # 尝试把i的for循环和k的for循环合并
"""
收缩压 0.0187891700183
舒张压 0.0217983962334
血清甘油三酯 0.106626654985
血清高密度脂蛋白 0.017607801794
血清低密度脂蛋白 0.0469415605048
0.0423527167071
收缩压 0.0187247433376
舒张压 0.0217903549985
血清甘油三酯 0.106435927756
血清高密度脂蛋白 0.0176354172964
血清低密度脂蛋白 0.0469157891125
0.0423004465003
收缩压 0.0187247433376
舒张压 0.0217561127815
血清甘油三酯 0.106435927756
血清高密度脂蛋白 0.0175999397542
血清低密度脂蛋白 0.0469157891125
0.0422865025485
只用几列特征
收缩压 0.0189131186035
舒张压 0.0220499115161
血清甘油三酯 0.107887773326
血清高密度脂蛋白 0.0180958828057
血清低密度脂蛋白 0.0484518547889
0.043079708208
收缩压 0.0188518551467
舒张压 0.0219142578067
血清甘油三酯 0.106245611968
血清高密度脂蛋白 0.0176502261518
血清低密度脂蛋白 0.0471239440939
0.0423571790334
数值列加的多了，反而过拟合严重了

加了一列2402之后
收缩压 0.0188815680407
舒张压 0.0218764386175
血清甘油三酯 0.106787708157
血清高密度脂蛋白 0.0178160315387
血清低密度脂蛋白 0.0470790390956
0.0424881570899


'0105', '0106', '0107', '0108', '0109', '0111', '0112', '019001', '019002', '019003', '019004', '019007',
'019008', '019017', '019032', '019033', '019034', '019035', '019036', '019037', '019038', '019039', '019040', '019041',
'019042', '019043', '019044', '019045', '019046', '019047', '019048', '019049', '019050', '019051', '019052', '019053',
'019054', '019055', '019056', '019059', '019062', '069002', '069003', '069004', '069005', '069007', '069008', '069010',
'069023', '069044', '069049', '069050', '1', '10000', '100005', '100007', '10005', '10012', '10014', '1123', '1124',
'1125', '1136', '129056', '129057', '129058', '129079', '131', '134', '1343', '1346', '1349', '1359', '137', '1456',
'1461', '1471', '159053', '159063', '179176', '179177', '179178', '179226', '1816', '184', '1849', '186', '1915',
'1918', '199118', '20000', '21A002', '21A012', '21A021', '229080', '2392', '2451', '2452', '2453', '2454', '269028',
'269029', '269030', '269031', '269052', '269055', '269056', '269057', '269058', '279001', '279002', '279003', '279004',
'279005', '279006', '279028', '299168', '300003', '300006', '300007', '300015', '300022', '300037', '300038', '300039',
'300069', '300070', '300072', '300075', '300077', '300080', '300087', '300111', '300114', '300117', '300118', '300119',
'300129', '300136', '300146', '300166', '300168', '300169', '300170', '300171', '300172', '300174', '300175', '300176',
'300178', '300179', '300180', '300181', '300182', '300183', '300184', '300185', '300186', '300187', '300188', '300189',
'300190', '31', '310', '311', '315', '316', '317', '3184', '319', '319100', '319159', '319273', '3205', '3206', '3211',
'3217', '33', '3302', '339105', '339106', '339107', '339114', '339122', '339125', '339126', '339128', '339129', '339130',
'339131', '339135', '34', '346', '35', '36', '369007', '369008', '369085', '369098', '369108', '37', '378', '3814',
'3816', '3818', '39', '419008', '439011', '439015', '439016', '439035', '459116', '459117', '459141', '459154', '459155',
'459156', '459158', '459159', '459161', '459181', '459182', '459183', '459184', '459206', '459207', '459208', '459209',
'459211', '459327', '459329', '459330', '459331', '459332', '459333', '459336', '459337', '459338', '459340', '459342',
'509006', '509013', '539004', '559007', '559046', '559047', '669001', '669004', '669005', '669010', '669014', '669043',
'669044', '669045', '669046', '699001', '699003', '699004', '699005', '699006', '699009', '709004', '709013', '709016',
'709019', '709020', '709022', '709023', '709024', '709025', '709027', '709030', '709031', '709043', '709044', '709048',
'729028', '739005', '759001', '769008', '769019', '809004', '809005', '809006', '809007', '809008', '809009', '809010',
'809011', '809012', '809013', '809014', '809015', '809016', '809017', '809018', '809019', '809020', '809022', '809023',
'809026', '809027', '809028', '809029', '809030', '809031', '809032', '809033', '809034', '809035', '809036', '809037',
'809038', '809039', '809040', '809041', '809042', '809043', '809044', '809045', '809046', '809047', '809048', '809049',
'809050', '809051', '809053', '809054', '809055', '809056', '809057', '809058', '809059', '809060', '809061', '819006',
'819008', '819009', '819010', '819011', '819012', '819013', '819014', '819015', '819016', '819017', '819019', '819020',
'819021', '819022', '819023', '819024', '819025', '819026', '819027', '819028', '819029', '819030', '819031', '839018',
'899021', '899022', '909001', '979010', '979024', '979025', '979026', '979027', '979029', '979091', '979092', '979093',
'979094', '979095', '979096', '989001', '989002', '989003', '989043', '989065', 'A49018', 'C19103', 'C39002', 'D29008',
'D29009', 'D29010', 'D29011', 'G49050', 'I19027', 'I69003', 'I69004', 'I69005', 'J29018', 'K29002', 'L19008', 'P19033',
'P79002', 'Q99001', 'Q99002', 'T99001', 'T99002', 'U99009', 'X19001', 'X19002', 'X19003', 'X19011', 'Y79001',

train_set_4
收缩压 0.0189317113558
舒张压 0.0220317515899
血清甘油三酯 0.107971005082
血清高密度脂蛋白 0.0182882229631
血清低密度脂蛋白 0.0487624020782
0.0431970186137
train_set_5
收缩压 0.0189413181362
舒张压 0.0220441078643
血清甘油三酯 0.107719402793
血清高密度脂蛋白 0.0183222016403
血清低密度脂蛋白 0.0487971658527
0.0431648392573
所有特征列之后的结果
收缩压 0.0189050175022
舒张压 0.0218541438658
血清甘油三酯 0.106317249944
血清高密度脂蛋白 0.0177675396663
血清低密度脂蛋白 0.0470901859208
0.0423868273798

'004997', '2403', '2404', '2405', '1814', '1815', '190', '191', '1840', '192', '1850', '10004',
'1117', '193', '314', '1115', '183', '2174', '10002', '10003', '31', '315',
'316', '319', '38', '312', '32', '2333', '100006', '313', '320', '1845',
'2372', '1845', '320', '100007', '2406', '100005', '37', '317', '33', '34',
'39', '300007', '1345', '2420', '155', '360', '1127', '269020', '269018', '269004',
'269023', '269016', '269005', '269007', '269024', '269022', '269021', '269015', '269025', '269013',
'269011', '269003', '269017', '269006', '269012', '269008', '269009', '269010', '269019', '269014',
'979012', '979021', '979016', '979004', '979007', '979020', '979017', '979008', '979011',
'979009', '979003', '979015', '979023', '979022', '979001', '979014', '979013', '979018',
'979019', '979002', '979006', '979005', '300021', '300019', '1106', '269014', '269016',
'139', '809001', '1107', '300018', '143', '669002', '669001', '1474', '669006', '300017',
'100014', '809009', '809008', '100012', '1112', '669005', '809004', '2386', '100013',
'2177', '809017', '669009', '300092', '809025', '809026', '809023', '809021', '10009',
'300008', '300001', '2409', '2376', '300013', '300012', '300011', '669004', '669021', '809010', '2302'             
"""