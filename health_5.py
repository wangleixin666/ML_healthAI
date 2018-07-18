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
merge_part1_2 = pd.read_csv(r'D:/kaggle/health/merge_part1_2.csv',encoding='utf-8')
test=pd.read_csv(r'D:/kaggle/health/meinian_round1_test_a_20180409.csv',sep=',',encoding='gbk')

test_of_part=merge_part1_2[merge_part1_2['vid'].isin(test['vid'])]
del merge_part1_2
gc.collect()
print '_____1_____'

test=pd.merge(test,test_of_part,on='vid')
del test_of_part
gc.collect()
print '-----2-----'

test.to_csv(r'D:/kaggle/health/test_set_1.csv',index=False,encoding='utf-8')
# 没有去除大量缺失的数据
del test
gc.collect()
print('---------------Done---------------------')
