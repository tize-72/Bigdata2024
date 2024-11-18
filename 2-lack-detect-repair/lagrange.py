"""拉格朗日插值法"""

import numpy  as np
import matplotlib.pyplot as plt

x = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
y = [14, 7, 5, 1, 2, 6, 7, 8, 13, 20, 21]
plt.figure()
plt.scatter(x,y) #x的第0列绘制在横轴，x的第1列绘制在纵轴
plt.show()
def lagrange(x, y, num_points, x_test):
    # 所有的基函数值，每个元素代表一个基函数的值
    l = np.zeros(shape=(num_points, ))

    # 计算第k个基函数的值
    for k in range(num_points):
        # 乘法时必须先有一个值
        # 由于l[k]肯定会被至少乘n次，所以可以取1
        l[k] = 1
        # 计算第k个基函数中第k_个项（每一项：分子除以分母）
        for k_ in range(num_points):
            # 这里没搞清楚，书中公式上没有对k=k_时，即分母为0进行说明
            # 有些资料上显示k是不等于k_的
            if k != k_:
                # 基函数需要通过连乘得到
                l[k] = l[k]*(x_test-x[k_])/(x[k]-x[k_])
            else:
                pass 
    # 计算当前需要预测的x_test对应的y_test值        
    L = 0
    for i in range(num_points):
        # 求所有基函数值的和
        L += y[i]*l[i]
    return L

x_test = list(np.linspace(-3, 7, 50))
y_predict = [lagrange(x, y, len(x), x_i) for x_i in x_test]

plt.figure()
plt.scatter(x_test,y_predict) #x的第0列绘制在横轴，x的第1列绘制在纵轴
plt.show()