#!/usr/bin/env python    # -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import time
import gc
import sys
sys_new = reload(sys)
sys_new.setdefaultencoding('utf-8')

start_time=time.time()
# 读取数据
train=pd.read_csv(r'D:/kaggle/health/meinian_round1_train_20180408.csv',sep=',',encoding='gbk')
test=pd.read_csv(r'D:/kaggle/health/meinian_round1_test_b_20180505.csv',sep=',',encoding='gbk')
data_part1=pd.read_csv(r'D:/kaggle/health/meinian_round1_data_part1_20180408.txt',sep='$',encoding='utf-8')
data_part2=pd.read_csv(r'D:/kaggle/health/meinian_round1_data_part2_20180408.txt',sep='$',encoding='utf-8')

# data_part1和data_part2进行合并，并剔除掉与train、test不相关vid所在的行
part1_2 = pd.concat([data_part1,data_part2],axis=0)
# {0/'index', 1/'columns'}, default 0
part1_2 = pd.DataFrame(part1_2).sort_values('vid').reset_index(drop=True)
vid_set=pd.concat([train['vid'],test['vid']],axis=0)
vid_set=pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
part1_2=part1_2[part1_2['vid'].isin(vid_set['vid'])]


# 根据常识判断无用的'检查项'table_id，过滤掉无用的table_id
def filter_None(data):
    data=data[data['field_results']!='']
    data=data[data['field_results']!='未查']
    return data

part1_2=filter_None(part1_2)

# 重复数据的拼接操作
def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df

# 数据简单处理
# print(part1_2.shape)
vid_tabid_group = part1_2.groupby(['vid','table_id']).size().reset_index()
# print(vid_tabid_group.head())
# print(vid_tabid_group.shape)
#                      vid               table_id  0
# 0  000330ad1f424114719b7525f400660b     0101     1
# 1  000330ad1f424114719b7525f400660b     0102     3

# 重塑index用来去重,区分重复部分和唯一部分
print('------------------------------去重和组合-----------------------------')
vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']
vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0]>1]['new_index']

# print(vid_tabid_group_dup.head()) #000330ad1f424114719b7525f400660b_0102
part1_2['new_index'] = part1_2['vid'] + '_' + part1_2['table_id']

dup_part = part1_2[part1_2['new_index'].isin(list(vid_tabid_group_dup))]
dup_part = dup_part.sort_values(['vid','table_id'])
unique_part = part1_2[~part1_2['new_index'].isin(list(vid_tabid_group_dup))]
del part1_2
gc.collect()

part1_2_dup = dup_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
part1_2_dup.rename(columns={0:'field_results'},inplace=True)
part1_2_res = pd.concat([part1_2_dup,unique_part[['vid','table_id','field_results']]])

# 行列转换
print('--------------------------重新组织index和columns---------------------------')
merge_part1_2 = part1_2_res.pivot(index='vid',values='field_results',columns='table_id')
del part1_2_res
gc.collect()
print('--------------新的part1_2组合完毕----------')
# print(merge_part1_2.shape)
merge_part1_2.to_csv(r'D:/kaggle/health/merge_part1_2.csv',encoding='utf-8')
print(merge_part1_2.head())
del merge_part1_2
gc.collect()
