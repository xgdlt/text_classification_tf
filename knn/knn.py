# ----------------------------------Sklearn实现Knn------------------------------
# KNeighborsClassifier(n_neighbors=5, weights=’uniform’,
# algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’,
# metric_params=None, n_jobs=None, **kwargs)[source]

# --------------------------------------实例一-----------------------------------------------
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(0)  # 设置随机种子，设置后每次产生的随机数都一样

# 加载数据集
iris = datasets.load_iris()  # 样本是150*4二维数据，代表150个样本，每个样本有4个特征
X = iris.data
y = iris.target
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# print('训练集：',X_train,'测试集：',X_test)
# print('训练集',len(X_train),'测试集：',len(X_test))

# 训练分类器
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# 预测
y_predict = knn.predict(X_test)
probility = knn.predict_proba(X_test)  # 计算各样本基于概率的预测
print(probility)
# 计算准确率
score = knn.score(X_test, y_test, sample_weight=None)

print('预测值：', y_predict, '实际值：', y_test, '得分：', score)
