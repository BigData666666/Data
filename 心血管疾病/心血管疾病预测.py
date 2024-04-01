#第一步！导入我们需要的工具
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("心血管疾病数据集.csv",header=0)
# 统计训练集数据缺失率,统计缺失率>0的数据，并且按缺失率从大到小进行排序
miss_rate = data.apply(lambda x:sum(x.isnull())/len(x),axis=0)
miss_rate = pd.DataFrame(miss_rate,columns=['缺失率'])
miss_rate = miss_rate[miss_rate['缺失率'] > 0].sort_values('缺失率',ascending=False)
#去除不重要的列
data.drop("id",axis=1,inplace=True)
data.insert(1,'age_year',data['age']/365)
prediction_var = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo','cholesterol', 'gluc', 'smoke', 'alco', 'active']
train, test = train_test_split(data, test_size = 0.2)
train_X= train[prediction_var]
train_y= train.cardio
test_X = test[prediction_var]
test_y = test.cardio
# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_X)
X_test_scaled = scaler.transform(test_X)
age=50
gender=2
height=168
weight=62
ap_hi=110
ap_lo=80
cholesterol=1
gluc=1
smoke=0
alco=0
active=1

# 逻辑回归
model3 = LogisticRegression()
model3.fit(X_train_scaled, train_y)

# 预测测试集
y_pred3 = model3.predict(X_test_scaled)

# 计算并输出预测的患病概率
new_sample = np.array([[age, gender, height, weight, ap_hi, ap_lo,cholesterol, gluc, smoke, alco, active]])
new_sample_scaled = scaler.transform(new_sample)
probability_of_disease3 = model3.predict_proba(new_sample_scaled)[:, 1]
print("患病概率:", probability_of_disease3)

# 评估模型
accuracy3 = model3.score(X_test_scaled, test_y)
print("准确率:", accuracy3)
