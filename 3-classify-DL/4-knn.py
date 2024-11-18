"""KNN算法
首先使用Kmeans
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.font_manager as fm
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



def prepare_data_and_train():
    # 1. 读取CSV数据
    try:
        df = pd.read_csv('./Mall_Customers.csv')  
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None, None, None
    
    # 打印数据基本信息
    print("\n数据基本信息:")
    print(df.info())
    print("\n数据前5行:")
    print(df.head())
    
    # 2. 提取特征
    X = df.iloc[:, [2, 4]].values  # 第三列和第五列作为特征
    
    # 检查是否有缺失值
    if np.isnan(X).any():
        print("\n警告：数据中存在缺失值，将进行处理")
        X = np.nan_to_num(X, nan=np.nanmean(X))
    
    print("\n特征数据形状:", X.shape)
    print("\n特征数据前5行:")
    print(X[:5])
    
    # 3. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. 使用K-means进行聚类，得到三个类别的标签
    kmeans = KMeans(n_clusters=3, random_state=42)
    y = kmeans.fit_predict(X_scaled)
    
    # 打印类别分布
    print("\n类别分布:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"类别 {u}: {c} 个样本")
    
    # 5. 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 6. 创建和训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # 7. 预测
    y_pred = knn.predict(X_test)
    
    # 8. 模型评估
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 9. 可视化结果
    
    # 9.1 绘制原始数据分布和聚类结果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter, label='类别')
    plt.title('数据聚类结果 (3类)')
    plt.xlabel('特征1 (标准化后)')
    plt.ylabel('特征2 (标准化后)')
    plt.show()
    
    # 9.2 绘制决策边界
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    fig, ax = plt.subplots(figsize=(12, 8))
    X0, X1 = X_scaled[:, 0], X_scaled[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, knn, xx, yy, cmap='viridis', alpha=0.4)
    scatter = ax.scatter(X0, X1, c=y, cmap='viridis', edgecolors='black')  # 修正这里
    plt.colorbar(scatter, label='类别')
    plt.title('KNN分类决策边界')
    plt.xlabel('特征1 (标准化后)')
    plt.ylabel('特征2 (标准化后)')
    plt.show()
    
    # 9.3 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    # 9.4 不同K值的准确率分析
    k_range = range(1, 31)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'bo-')
    plt.title('不同K值的模型准确率')
    plt.xlabel('K值')
    plt.ylabel('准确率')
    plt.grid(True)
    plt.show()

    return knn, scaler, kmeans

def predict_new_sample(knn, scaler, kmeans, new_sample):
    """
    预测新样本的类别
    """
    new_sample_scaled = scaler.transform(new_sample.reshape(1, -1))
    prediction = knn.predict(new_sample_scaled)
    return prediction[0]

def main():
    # 运行分类
    knn, scaler, kmeans = prepare_data_and_train()
    
    if knn is not None:
        # 示例：预测新样本
        # new_sample = np.array([value1, value2])  
        # predicted_class = predict_new_sample(knn, scaler, kmeans, new_sample)
        # print(f"预测类别: {predicted_class}")
        pass

if __name__ == "__main__":
    main()