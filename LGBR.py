# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:02:42 2022

@author: NJ
"""


from lightgbm import LGBMRegressor as lgb
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold,cross_validate as CVS,train_test_split as TTS
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# 载入数据集
data=pd.read_csv("E:\SHUIJU\senfgas 5.csv")
data=data.dropna(axis=0)
#改变采样率，预测效果的对比
data=data.sample(frac=0.9) #抽取百分之2的数据


##划分数据集
#X = data.iloc[:,np.r_[2,3,4,5,6,7,8,9,11,12,13,14]].values
x = data.iloc[:,1:7]
X = x.values.astype(np.float64)
y =data.iloc[:,13]
Y = y.values.astype(np.float64)

#划分数据集
X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.3, random_state=42)
'''
X_MinMax= MinMaxScaler(feature_range=(0,1))
X=X_MinMax.fit_transform(X)
'''

transfer=StandardScaler()
X_train=transfer.fit_transform(X_train)
X_test=transfer.transform(X_test)

model = lgb(objective='regression'
                     ,num_leaves=1050
                     ,n_estimators=1500
                     #,maX_bin=1000
                     ,learning_rate=0.05      
                     ,max_depth=-1
                     ,force_col_wise=True
                     ,colsample_bytree=0.8
                     ,subsample_for_bin=220000
                     ,random_state=100#随机数种子，默认无
                     ,importance_type='split'
                     ,n_jobs= -1
                     ).fit(X_train, y_train)


Y_predict1 = model.predict(X_train)
Y_predict2 = model.predict(X_test)



MSE=metrics.mean_squared_error(y_train,Y_predict1)
print('MSE_train={}'.format(MSE))
MAE=metrics.mean_absolute_error(y_train,Y_predict1)
print('MAE_train={}'.format(MAE))
RMSE=np.sqrt(metrics.mean_squared_error(y_train,Y_predict1))  # RMSE
print('RMSE_train={}'.format(RMSE))
R2=metrics.r2_score(y_train,Y_predict1)
print('R2_train={}'.format(R2))


MSE=metrics.mean_squared_error(y_test,Y_predict2)
print('MSE_test={}'.format(MSE))
MAE=metrics.mean_absolute_error(y_test,Y_predict2)
print('MAE_test={}'.format(MAE))
RMSE=np.sqrt(metrics.mean_squared_error(y_test,Y_predict2))  # RMSE
print('RMSE_test={}'.format(RMSE))
R2=metrics.r2_score(y_test,Y_predict2)
print('R2_test={}'.format(R2))


cv = KFold(n_splits=5 ,shuffle=True,random_state=1402)

scores= CVS(model,X,Y,cv=cv,return_train_score=True
                   ,scoring=('r2','neg_mean_absolute_error', 'neg_mean_squared_error')
                   ,verbose=True,n_jobs=-1)
    
importances = model.feature_importances_
ax = plt.scatter(y_test,Y_predict2)
plt.scatter(Y_predict1,y_train)
plt.scatter(Y_predict2,y_test)

a= model.feature_importances_
b=a*100
print(b)
