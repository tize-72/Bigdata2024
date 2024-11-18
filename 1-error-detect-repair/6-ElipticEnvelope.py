"""鲁棒的协方差估计"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope


np.random.seed(1)
random_data = np.random.randn(500,2)  * 20 + 20
random_data_outliers = np.random.randn(100,2)  * 30 + 50
# 合并
random_total  = np.concatenate((random_data,random_data_outliers),axis=0)
plt.scatter(random_total[:,0],random_total[:,1]) #x的第0列绘制在横轴，x的第1列绘制在纵轴
plt.show()

model = EllipticEnvelope(contamination=.1)
preds = model.fit_predict(random_total)
outliers, outliers_idx = [], []
for idx, item in enumerate(preds):
    if item == -1:
        outliers.append(random_total[idx])
        outliers_idx.append(idx)

#可视化预测结果
plt.scatter(random_total[:,0],random_total[:,1],c=preds)  #样本点的颜色由y值决定
plt.show()