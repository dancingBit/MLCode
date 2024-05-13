# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:34:05 2024

@author: NJ
"""

from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,cross_validate as CVS
# 载入数据集
dataset = pd.read_csv(r"E:\SHUIJU\senfgas 5.csv",encoding='unicode_escape')
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
#X = dataset[:,0:5]
#Y = dataset[:,6]
X = dataset.iloc[:, 1:7].values  
Y = dataset.iloc[:,  13].values



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1314)



'''

transfer=StandardScaler()
X_train=transfer.fit_transform(X_train)
X_test=transfer.transform(X_test)

'''

model = XGBRegressor( 
                         learning_rate=0.1, 
                         n_estimators=250, 
                        # max_depth=4,
                        # min_child_weight=4,
                          subsample=0.8,
                         gamma=5,
                         colsample_bytree=0.9,
                          n_jobs=-2, 
                        # nthread=None,            
                         reg_alpha=4, 
                        # reg_lambda=4,
                         scale_pos_weight=7.2, 
                        # base_score=4.0,   
                         # missing=None,
                         importance_type='total_gain')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_predion = model.predict(X_train)
predictions = [round(value) for value in y_pred]

MSE=metrics.mean_squared_error(y_train,y_predion)
print('MSE={}'.format(MSE))
MAE=metrics.mean_absolute_error(y_train,y_predion)
print('MAE={}'.format(MAE))
RMSE=np.sqrt(metrics.mean_squared_error(y_train,y_predion))  # RMSE,sqrt返回平方根
print('RMSE={}'.format(RMSE))

MSE=metrics.mean_squared_error(y_test,y_pred)
print('MSE={}'.format(MSE))
MAE=metrics.mean_absolute_error(y_test,y_pred)
print('MAE={}'.format(MAE))
RMSE=np.sqrt(metrics.mean_squared_error(y_test,y_pred))  # RMSE,sqrt返回平方根
print('RMSE={}'.format(RMSE))




R2=metrics.r2_score(y_test,y_pred)
R21=metrics.r2_score(y_train,y_predion)
print('R2={}'.format(R2))
print('R2={}'.format(R21))
import matplotlib.pyplot as plt
ax = plt.scatter(y_test,y_pred)

cv = KFold(n_splits=5 ,shuffle=True,random_state=1402)

scores= CVS(model,X,Y,cv=cv,return_train_score=True
                   ,scoring=('r2','neg_mean_absolute_error', 'neg_mean_squared_error')
                   ,verbose=True,n_jobs=-1)

importances = XGBRegressor.feature_importances_

a=model.feature_importances_
b=a*100
print(b)

#b=a*100
#print(b)
