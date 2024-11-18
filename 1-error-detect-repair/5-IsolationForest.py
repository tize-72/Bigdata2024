"""孤立森林"""

from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

# multiply and add by random numbers to get some real values
np.random.seed(1)
random_data = np.random.randn(500,2)  * 20 + 20
random_data_outliers = np.random.randn(100,2)  * 30 + 50
# 合并
random_total  = np.concatenate((random_data,random_data_outliers),axis=0)
plt.scatter(random_total[:,0],random_total[:,1]) #x的第0列绘制在横轴，x的第1列绘制在纵轴
plt.show()

# 调用包生成模型
model = IsolationForest(max_samples=100,contamination= 'auto')
outliers, outliers_idx = [], []
# 利用模型进行预测，检测
preds = model.fit_predict(random_total)

for idx, item in enumerate(preds):
    if item == -1:
        outliers.append(random_total[idx])
        outliers_idx.append(idx)

colors = []
for item in preds:
    if item == -1:
        colors.append('r')
        continue
    colors.append('b')
#可视化预测结果
plt.scatter(random_total[:,0],random_total[:,1],c=colors)  #样本点的颜色由y值决定
plt.show()