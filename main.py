import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 获取当前工作目录
origin_path = os.getcwd()
file_path = origin_path+"/letterecognition.csv"
data = pd.read_csv(file_path)

# 判断有无缺失值，当前数据集无缺失值，所以不进行填充数据缺失等操作
if data.isnull().sum().sum() == 0:
    print("无缺失值.")
else:
    print("以下列存在缺失值:", data.isnull().sum())

# 进行数值编码，将字符型数据转换为数值型数据，但是对实际的提升不明显
# label_encoder = LabelEncoder()
# data['encoded_letter'] = label_encoder.fit_transform(data['letter'])

#  划分特征和标签
X = data.drop(['letter'], axis=1)
y = data['letter']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

"""数据标准化：通过标准化处理，调整所有特征的均值为0和标准差为1
经过测试，发现标准化后的模型准确率略有提升，标准化前的模型准确率为0.75-0.76，标准化后为0.77-0.78
标准化后的模型准确率提升并不明显，可能是因为数据集本身的数值分布就比较均衡。
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用标准化后的数据进行模型训练(逻辑回归部分)
logi_model = LogisticRegression()
logi_model.fit(X_train_scaled, y_train)
y_train_pred_logi = logi_model.predict(X_train_scaled)
y_test_pred_logi = logi_model.predict(X_test_scaled)

# 计算准确率
logi_train_accuracy = accuracy_score(y_train, y_train_pred_logi)
logi_test_accuracy = accuracy_score(y_test, y_test_pred_logi)

# 输出使用逻辑回归模型的准确率
print(f"训练集准确率: {logi_train_accuracy:.2f}")
print(f"测试集准确率: {logi_test_accuracy:.2f}")
print(classification_report(y_test, y_test_pred_logi))

# 绘制准确率对比图
acc_data = {'Train': logi_train_accuracy, 'Test': logi_test_accuracy}
fig, ax = plt.subplots()
ax.bar(acc_data.keys(), acc_data.values(), color=['blue', 'green'])
ax.set_ylabel('Accuracy')
ax.set_title('Logistic Regression Model Accuracy: Train vs. Test')
plt.show()

# 使用SVM模型来进行训练
# 这里的SVC是SVM模型下面是用于分类任务的一个实现方法
# 然后使用了rbf核函数，这个核函数是用于处理高维数据的一种函数，而光学文本分类是非线性问题，所以使用该函数性能较好
SVM_model = SVC(kernel='rbf', C=1.0, random_state=7)
SVM_model.fit(X_train_scaled, y_train)
y_train_pred_SVM = SVM_model.predict(X_train_scaled)
y_test_pred_SVM = SVM_model.predict(X_test_scaled)

# 计算准确率
SVM_train_accuracy = accuracy_score(y_train, y_train_pred_SVM)
SVM_test_accuracy = accuracy_score(y_test, y_test_pred_SVM)

# 输出使用SVM模型的准确率
print(f"训练集准确率: {SVM_train_accuracy:.2f}")
print(f"测试集准确率: {SVM_test_accuracy:.2f}")
print(classification_report(y_test, y_test_pred_SVM))

# 绘制准确率对比图
acc_data = {'Train': SVM_train_accuracy, 'Test': SVM_test_accuracy}
fig, ax = plt.subplots()
ax.bar(acc_data.keys(), acc_data.values(), color=['blue', 'green'])
ax.set_ylabel('Accuracy')
ax.set_title('SVM Model Accuracy: Train vs. Test')
plt.show()

# 绘制SVM和逻辑回归在测试集上的准确率对比图，可以看出，SVM的准确率高于逻辑回归，达到了94%，高了逻辑回归16%
# 说明SVM在这个数据集中的光学字符分类效果优于逻辑回归
against_data = {'SVM Test': SVM_test_accuracy, 'Logistic Regression Test': logi_test_accuracy}
fig, ax = plt.subplots()
ax.bar(against_data.keys(), against_data.values(), color=['blue', 'green'])
ax.set_ylabel('Accuracy')
ax.set_title('Logistic Regression vs. SVM Test Set Accuracy')
plt.show()