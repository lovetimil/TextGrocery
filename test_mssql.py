# coding:utf-8
#!/usr/bin/evn python
# train and predict, based on validation params
import os
from mssql import MSSQL
from tgrocery import Grocery 
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

	    

#
ms = MSSQL(host="192.168.1.200",user="xiehuabo",pwd="xhb123",db="tt_highlights_news")
#reslist = ms.ExecQuery('''select nid,cid1,htmlcontent from (select nid,cid1,
#	htmlcontent,ROW_NUMBER() over(PARTITION by cid1 order by nid ) as num 
#	from tt_highlights _news.dbo.news where cid1<21  and cid1 >9 
#	and datalength(htmlcontent)>100) T where T.num < 625'''
#)
reslist = ms.ExecQuery('''select nid,cid1,htmlcontent from 
( 
select nid,cid1,htmlcontent,ROW_NUMBER() over(PARTITION by cid1 order by nid ) as num from tt_highlights_news.dbo.news where cid1<21  and cid1 >9 and datalength(htmlcontent)>100
) T where T.num < 625''')
dic={'id':[],'type':[],'contents':[]}
tdic={'id':[],'type':[],'contents':[]}
i = 0
pre_type = -1
for _id,_type,contents in reslist:

	if (i%624) < 563:
#	if (i%25) < 20:
		dic['id'].append(_id)
		dic['type'].append(_type)
		dic['contents'].append(contents)
	else :
		tdic['id'].append(_id)
		tdic['type'].append(_type)
		tdic['contents'].append(contents)
	i +=1
	
#train = pd.read_csv( train_file, header = 0, delimiter = "\t", quoting = 3 )
#test = pd.read_csv( test_file, header = 1, delimiter = "\t", quoting = 3 )
train = DataFrame(dic)
test = DataFrame(tdic)
#
#classfynews_instance 是模型保存路径
grocery = Grocery('classfynews_instance')

train_in = [train['contents'],train['type']]
grocery.train(train_in)
print grocery.get_load_status()
#grocery.save()

copy_grocery = Grocery('classfynews_instance')
copy_grocery.load()
#copy_grocery = grocery
test_in = [test['contents'],test['type']]
#输入类似 ['我是中国人','台北*****']
#输出 [11,12]
test_result = copy_grocery.predict(test['contents'])
print test_result.predicted_y
#test_result = copy_grocery.test(test_in)
#print test_result.show_result()


