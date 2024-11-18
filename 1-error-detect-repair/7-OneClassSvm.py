"""一类支持向量机"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

np.random.seed(1)
random_data = np.random.randn(500,2)  * 20 + 20
random_data_outliers = np.random.randn(100,2)  * 30 + 50
# 合并
random_total  = np.concatenate((random_data,random_data_outliers),axis=0)
plt.scatter(random_total[:,0],random_total[:,1]) #x的第0列绘制在横轴，x的第1列绘制在纵轴
plt.show()

model = svm.OneClassSVM(kernel="rbf",
                         gamma='scale', 
                        coef0=0.0, 
                        tol=0.001, 
                        nu=0.5, 
                        shrinking=True, 
                        cache_size=200, 
                        verbose=False, 
                        max_iter=-1)
preds = model.fit_predict(random_total)
outliers, outliers_idx = [], []


for idx, item in enumerate(preds):
    if item == -1:
        outliers.append(random_total[idx])
        outliers_idx.append(idx)

colors = []
for idx, item in enumerate(preds):
    if item == -1:
        colors.append('r')
        continue
    colors.append('b')

#可视化预测结果
plt.scatter(random_total[:,0],random_total[:,1],c=colors)  #样本点的颜色由y值决定
plt.show()