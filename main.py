import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

# 使用标准化后的数据进行模型训练
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

#  输出模型准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
print(classification_report(y_test, y_pred))