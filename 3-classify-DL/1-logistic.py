"""逻辑回归判别法"""


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import datasets


#导入数据
iris = datasets.load_iris()
#区分数据的自变量和因变量
iris_X = iris.data
iris_Y = iris.target
#将数据分成训练集和测试集，比例为：80%和20%
iris_train_X , iris_test_X, iris_train_Y ,iris_test_Y = train_test_split(
        iris_X, iris_Y, test_size=0.2,random_state=0)
#训练逻辑回归模型

model = LogisticRegression() #此处这个函数中有很多参数可供选择
model.fit(iris_train_X, iris_train_Y)


#预测
predict = model.predict(iris_test_X)
accuracy = model.score(iris_test_X,iris_test_Y)

print(predict)
print(accuracy)