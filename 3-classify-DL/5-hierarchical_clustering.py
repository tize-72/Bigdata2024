"""层次聚类算法"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data(file_path, feature_columns):
    """
    加载和预处理数据
    
    参数:
        file_path (str): 数据文件路径
        feature_columns (list): 用于聚类的特征列索引
    返回:
        tuple: (标准化后的数据, 原始数据, 特征名称)
    """
    # 读取数据
    df = pd.read_csv(file_path)
    print("数据前5行：")
    print(df.head())
    
    # 提取特征
    X = df.iloc[:, feature_columns].values
    feature_names = df.columns[feature_columns].tolist()
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X, feature_names

def plot_dendrogram(X, method='ward', metric='euclidean'):
    """
    绘制层次聚类树状图
    
    参数:
        X (array): 输入数据
        method (str): 链接方法
        metric (str): 距离度量方法
    """
    plt.figure(figsize=(12, 8))
    
    # 计算链接矩阵
    linkage_matrix = linkage(X, method=method, metric=metric)
    
    # 绘制树状图
    dendrogram(linkage_matrix)
    plt.title('层次聚类树状图')
    plt.xlabel('样本索引')
    plt.ylabel('距离')
    plt.axhline(y=np.mean(linkage_matrix[:, 2]), color='r', linestyle='--', 
                label='建议切分线')
    plt.legend()
    plt.show()

def perform_hierarchical_clustering(X, n_clusters=2, method='ward', metric='euclidean'):
    """
    执行层次聚类
    
    参数:
        X (array): 输入数据
        n_clusters (int): 聚类数量
        method (str): 链接方法
        metric (str): 距离度量方法
    返回:
        array: 聚类标签
    """
    # 计算链接矩阵
    linkage_matrix = linkage(X, method=method, metric=metric)
    
    # 获取聚类标签
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    
    return labels

def visualize_clusters(X, labels, feature_names, X_original=None):
    """
    可视化聚类结果
    
    参数:
        X (array): 标准化后的数据
        labels (array): 聚类标签
        feature_names (list): 特征名称
        X_original (array): 原始数据
    """
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 标准化数据的散点图
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax1.set_title('标准化数据的聚类结果')
    ax1.set_xlabel(f'标准化 {feature_names[0]}')
    ax1.set_ylabel(f'标准化 {feature_names[1]}')
    plt.colorbar(scatter1, ax=ax1, label='聚类类别')
    
    # 2. 原始数据的散点图（如果提供）
    if X_original is not None:
        scatter2 = ax2.scatter(X_original[:, 0], X_original[:, 1], c=labels, cmap='viridis')
        ax2.set_title('原始数据的聚类结果')
        ax2.set_xlabel(feature_names[0])
        ax2.set_ylabel(feature_names[1])
        plt.colorbar(scatter2, ax=ax2, label='聚类类别')
    
    # 3. 箱线图
    cluster_df = pd.DataFrame({
        feature_names[0]: X_original[:, 0] if X_original is not None else X[:, 0],
        feature_names[1]: X_original[:, 1] if X_original is not None else X[:, 1],
        '聚类': labels
    })
    
    sns.boxplot(x='聚类', y=feature_names[0], data=cluster_df, ax=ax3)
    ax3.set_title(f'{feature_names[0]}的分布')
    
    sns.boxplot(x='聚类', y=feature_names[1], data=cluster_df, ax=ax4)
    ax4.set_title(f'{feature_names[1]}的分布')
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 1. 加载和预处理数据
    file_path = './Mall_Customers.csv'  
    feature_columns = [2, 4]     
    X_scaled, X_original, feature_names = load_and_prepare_data(file_path, feature_columns)
    
    # 2. 绘制树状图
    plot_dendrogram(X_scaled)
    
    # 3. 执行层次聚类（指定2类）
    labels = perform_hierarchical_clustering(X_scaled, n_clusters=2)
    
    # 4. 可视化聚类结果
    visualize_clusters(X_scaled, labels, feature_names, X_original)
    

if __name__ == "__main__":
    main()