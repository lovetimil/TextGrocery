# coding:utf-8
#!/usr/bin/evn python
from tgrocery import Grocery 


copy_grocery = Grocery('./classfynews_instance')#模型所在路径
copy_grocery.load()
#copy_grocery = grocery
test = ['我是中国人','台北*****']
test_result = copy_grocery.predict(test)
print test_result.predicted_y
#test_result = copy_grocery.test(test_in)
#print test_result.show_result()


