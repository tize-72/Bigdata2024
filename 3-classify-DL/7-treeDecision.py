"""决策树算法"""

# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 将数据转换为DataFrame以便更好地查看
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. 创建和训练决策树模型
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# 4. 进行预测
y_pred = dt.predict(X_test)

# 5. 评估模型
print("模型准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred, 
                          target_names=iris.target_names))

# 6. 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt.feature_importances_
})
print("\n特征重要性:")
print(feature_importance.sort_values('importance', ascending=False))

# 7. 可视化决策树
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          filled=True, rounded=True)
plt.title("决策树可视化")
plt.show()

# 8. 可视化特征重要性
plt.figure(figsize=(10,6))
sns.barplot(data=feature_importance.sort_values('importance', ascending=False), 
            x='feature', y='importance')
plt.title("特征重要性可视化")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. 数据分布可视化
plt.figure(figsize=(12,8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2,2,i+1)
    for target in range(3):
        plt.hist(df[df['target']==target][feature], 
                label=iris.target_names[target], alpha=0.7)
    plt.title(feature)
    plt.legend()
plt.tight_layout()
plt.show()