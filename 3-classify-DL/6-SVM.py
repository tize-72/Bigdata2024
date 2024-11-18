"""支持向量机"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子，保证结果可复现
np.random.seed(42)


def generate_data(n_samples=300):
    """
    生成非线性可分的二分类数据
    
    参数:
        n_samples: 样本数量
    返回:
        X: 特征数据
        y: 标签
    """
    # 生成第一类数据：环形数据
    n1 = n_samples // 2
    r1 = np.random.normal(0.5, 0.1, n1)
    theta1 = np.random.uniform(0, 2*np.pi, n1)
    X1 = np.column_stack([
        r1 * np.cos(theta1),
        r1 * np.sin(theta1)
    ])
    y1 = np.zeros(n1)

    # 生成第二类数据：另一个环形数据
    n2 = n_samples - n1
    r2 = np.random.normal(1.0, 0.1, n2)
    theta2 = np.random.uniform(0, 2*np.pi, n2)
    X2 = np.column_stack([
        r2 * np.cos(theta2),
        r2 * np.sin(theta2)
    ])
    y2 = np.ones(n2)

    # 合并数据
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    return X, y

def plot_decision_boundary(X, y, model, title="决策边界可视化"):
    """
    绘制SVM决策边界
    
    参数:
        X: 特征数据
        y: 标签
        model: 训练好的SVM模型
        title: 图标题
    """
    plt.figure(figsize=(12, 5))
    
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.subplot(121)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    # 绘制支持向量
    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.scatter(X[model.support_, 0], X[model.support_, 1], 
               c='red', marker='x', s=200, linewidth=2, 
               label='支持向量')
    plt.title("支持向量展示")
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(y_true, y_pred):
    """
    评估模型性能
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
    """
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

def main():
    """主函数"""
    
    # 1. 生成数据
    print("正在生成数据...")
    X, y = generate_data(300)
    
    # 2. 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 3. 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 创建并训练SVM模型
    print("\n正在训练SVM模型...")
    # 使用RBF核函数，这适合非线性分类问题
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # 5. 模型预测
    print("\n进行预测...")
    y_pred = svm_model.predict(X_test_scaled)
    
    # 6. 模型评估
    print("\n模型评估结果：")
    train_score = svm_model.score(X_train_scaled, y_train)
    test_score = svm_model.score(X_test_scaled, y_test)
    print(f"训练集准确率: {train_score:.4f}")
    print(f"测试集准确率: {test_score:.4f}")
    
    # 7. 详细评估
    evaluate_model(y_test, y_pred)
    
    # 8. 决策边界可视化
    plot_decision_boundary(X_train_scaled, y_train, svm_model, 
                         "SVM决策边界 (训练集)")

if __name__ == "__main__":
    main()