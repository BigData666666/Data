#第一步！导入我们需要的工具
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("糖尿病数据集.csv",header=0)
# 统计训练集数据缺失率,统计缺失率>0的数据，并且按缺失率从大到小进行排序
miss_rate = data.apply(lambda x:sum(x.isnull())/len(x),axis=0)
miss_rate = pd.DataFrame(miss_rate,columns=['缺失率'])
miss_rate = miss_rate[miss_rate['缺失率'] > 0].sort_values('缺失率',ascending=False)
gender = {'Male':2,'Female':1}
data["Gender"] = data["Gender"].map(gender)
data.replace("Yes",1,inplace=True)
data.replace("No",0,inplace=True)
data.replace("Positive",1,inplace=True)
data.replace("Negative",0,inplace=True)
prediction_var = ["Age","Gender","Polyuria","Polydipsia","sudden weight loss","weakness","Polyphagia","Genital thrush","visual blurring","Itching","Irritability","delayed healing","partial paresis","muscle stiffness","Alopecia","Obesity"]
train, test = train_test_split(data, test_size = 0.3)
train_X= train[prediction_var]
train_y= train["class"]
test_X = test[prediction_var]
test_y = test["class"]
# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_X)
X_test_scaled = scaler.transform(test_X)
#随机森林
model1=RandomForestClassifier(n_estimators=71,max_features=5)
model1.fit(X_train_scaled, train_y)

# 预测测试集
y_pred1 = model1.predict(X_test_scaled)

Age=18
Gender=2
Polyuria=0
Polydipsia=1
sudden_weight_loss=0
weakness=1
Polyphagia=1
Genital_thrush=1
visual_blurring=0
Itching=1
Irritability=0
delayed_healing=1
partial_paresis=1
muscle_stiffness=1
Alopecia=1
Obesity=1
# 计算并输出预测的患病概率
new_sample = np.array([[Age,Gender,Polyuria,Polydipsia,sudden_weight_loss,weakness,Polyphagia,Genital_thrush,visual_blurring,Itching,Irritability,delayed_healing,partial_paresis,muscle_stiffness,Alopecia,Obesity]])
new_sample_scaled=scaler.transform(new_sample)
probability_of_disease1 = model1.predict_proba(new_sample_scaled)[:, 1]
print("患病概率:", probability_of_disease1)
