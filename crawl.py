import numpy as np
import collections
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 创建模拟数据

# 生成随机的特征数据，例如薪资、工作经验、地区、教育背景
num_samples = 10000
salary = np.random.uniform(30000, 100000, num_samples)
experience = np.random.uniform(0, 20, num_samples)

# 随机生成地区数据（0表示城市A，1表示城市B）
region = np.random.randint(0, 2, num_samples)

# 随机生成教育背景数据（0表示本科，1表示硕士）
education = np.random.randint(0, 2, num_samples)

# 创建一个随机的目标变量（0或1）
# 假设如果薪资大于60000、工作经验大于5年、地区为城市A、教育背景为硕士，则目标变量为1，否则为0
target = np.where((salary > 60000) & (experience > 5) & (region == 0) & (education == 1), 1, 0)
print(collections.Counter(target))
# 将特征数据合并成一个特征矩阵
X = np.column_stack((salary, experience, region, education))
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# 创建并训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# 输出模型性能指标
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# with open('clf.pickle','wb') as f:
#     pickle.dump(model,f) #将训练好的模型clf存储在变量f中，且保存到本地

with open('clf.pickle', 'rb') as f:
    clf_load = pickle.load(f)  #将模型存储在变量clf_load中
    print(clf_load.predict([[90000, 1, 0, 1]])) #调用模型并预测结果