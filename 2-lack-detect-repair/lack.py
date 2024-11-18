"""结构数据的缺失值检测与诊断"""

# 导入第三方包
import pandas as pd

# 读入外部数据
data_excel = pd.read_excel('./data.xlsx') # 如果报文件路径错误，就替换为绝对路径
# 查看数据的规模
data_excel.shape
# 数值型转字符型
data_excel['id'] = data_excel['id'].astype(str)
# 字符型转数值型
data_excel['custom_amt'] = data_excel['custom_amt'].str[1:].astype(float)
# 字符型转日期型
data_excel['order_date'] = pd.to_datetime(data_excel['order_date'], format = '%Y年%m月%d日')
# 重新查看数据集的各变量类型
data_excel.dtypes
# 预览数据的前5行
data_excel.head()
# 判断数据中是否存在重复观测
data_excel.duplicated().any()

# 构造数据
df = pd.DataFrame(dict(name = ['张三','李四','王二','张三','赵五','丁一','王二'],
                      gender = ['男','男','女','男','女','女','男'],
                      age = [29,25,27,29,21,22,27],
                      income = [15600,14000,18500,15600,10500,18000,13000],
                      edu = ['本科','本科','硕士','本科','大专','本科','硕士']))
# 查看数据
df
# 判断数据中是否存在重复观测
df.duplicated().any()
df.drop_duplicates(subset=['name','age'])

# 判断各变量中是否存在缺失值
data_excel.isnull().any(axis = 0)
# 各变量中缺失值的数量
data_excel.isnull().sum(axis = 0)
# 各变量中缺失值的比例
data_excel.isnull().sum(axis = 0)/data_excel.shape[0]
# 删除字段 -- 如删除缺失率非常高的edu变量
data_excel.drop(labels = 'edu', axis = 1, inplace=True)
# 数据预览
data_excel.head()
# 删除观测，-- 如删除age变量中所对应的缺失观测
data_excel_new = data_excel.drop(labels = data_excel.index[data_excel['age'].isnull()], axis = 0)
# 查看数据的规模
data_excel_new.shape