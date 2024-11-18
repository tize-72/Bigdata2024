"""Kmeans聚类"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys

# 设置中文字体，这个字体需要自己安装，Ubuntu系统需要搜索安装方法，Windows直接交互安装即可
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data(file_path, feature_columns):
    """
    加载和预处理数据
    
    参数:
    file_path: CSV文件路径
    feature_columns: 用于聚类的特征列索引列表
    
    返回:
    X_scaled: 标准化后的特征数据
    X: 原始特征数据
    """
    # 读取数据
    try:
        df = pd.read_csv(file_path)
        print("数据加载成功！")
        print("\n数据基本信息：")
        print(df.info())
        print("\n数据前5行：")
        print(df.head())
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None

    # 提取特征
    X = df.iloc[:, feature_columns].values
    
    # 检查并处理缺失值
    if np.isnan(X).any():
        print("\n检测到缺失值，使用均值填充")
        X = np.nan_to_num(X, nan=np.nanmean(X))
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X

def find_optimal_clusters(X, max_clusters=10):
    """
    使用轮廓系数找到最优的聚类数
    
    参数:
    X: 标准化后的特征数据
    max_clusters: 最大尝试的聚类数
    
    返回:
    optimal_clusters: 最优聚类数
    silhouette_scores: 不同聚类数对应的轮廓系数
    """
    silhouette_scores = []
    K = range(2, max_clusters+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        print(f'聚类数 {k} 的轮廓系数: {score:.3f}')
    
    optimal_clusters = K[np.argmax(silhouette_scores)]
    
    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('聚类数')
    plt.ylabel('轮廓系数')
    plt.title('不同聚类数的轮廓系数变化')
    plt.grid(True)
    plt.show()
    
    return optimal_clusters, silhouette_scores

def perform_kmeans(X, n_clusters):
    """
    执行KMeans聚类
    
    参数:
    X: 标准化后的特征数据
    n_clusters: 聚类数
    
    返回:
    kmeans: 训练好的KMeans模型
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

def visualize_clusters(X, kmeans, X_original=None):
    """
    可视化聚类结果
    
    参数:
    X: 标准化后的特征数据
    kmeans: 训练好的KMeans模型
    X_original: 原始特征数据（可选）
    """
    # 1. 散点图展示聚类结果
    plt.figure(figsize=(12, 5))
    
    # 标准化数据的散点图
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.colorbar(scatter, label='聚类类别')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidth=3, label='聚类中心')
    plt.title('标准化数据的聚类结果')
    plt.xlabel('特征1 (标准化)')
    plt.ylabel('特征2 (标准化)')
    plt.legend()
    
    # 如果提供了原始数据，则显示原始数据的散点图
    if X_original is not None:
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_original[:, 0], X_original[:, 1], c=kmeans.labels_, cmap='viridis')
        plt.colorbar(scatter, label='聚类类别')
        plt.title('原始数据的聚类结果')
        plt.xlabel('特征1 (原始)')
        plt.ylabel('特征2 (原始)')
    
    plt.tight_layout()
    plt.show()
    
    # 2. 展示每个类别的样本数量分布
    unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
    plt.figure(figsize=(8, 6))
    plt.bar(unique_labels, counts)
    plt.title('各聚类类别的样本数量分布')
    plt.xlabel('聚类类别')
    plt.ylabel('样本数量')
    plt.show()


def main():
    # 1. 加载和预处理数据
    file_path = './Mall_Customers.csv'  # CSV文件路径，找不到的话就替换为绝对路径
    feature_columns = [2, 4]     # 特征列索引
    X_scaled, X_original = load_and_prepare_data(file_path, feature_columns)
    
    if X_scaled is None:
        return
    
    # 2. 寻找最优聚类数
    optimal_clusters, _ = find_optimal_clusters(X_scaled)
    print(f"\n最优聚类数: {optimal_clusters}")
    
    # 3. 执行KMeans聚类
    kmeans = perform_kmeans(X_scaled, optimal_clusters)
    
    # 4. 可视化结果
    visualize_clusters(X_scaled, kmeans, X_original)

if __name__ == "__main__":
    main()