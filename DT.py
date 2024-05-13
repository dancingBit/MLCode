# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:37:32 2024

@author: NJ
"""


import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold,cross_validate as CVS

dataset = pd.read_csv("E:\SHUIJU\senfgas 5.csv", encoding='unicode_escape')
dataset = shuffle(dataset)
X = dataset.iloc[:, 1:7] 
y = dataset.iloc[:,13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


regr_1 = DecisionTreeRegressor(splitter = 'best', 
                               random_state=830, 
                               min_samples_split=2, 
                               min_samples_leaf=4, 
                               max_depth=10)

regr_1.fit(X_train, y_train)
y_1 = regr_1.predict(X_train)
y_2 = regr_1.predict(X_test)




score =regr_1.score(X_train, y_train)
RMSE = np.sqrt(metrics.mean_squared_error(y_train, y_1))
MAE = metrics.mean_absolute_error(y_train, y_1)
R2 = r2_score(y_train, y_1)
#print(f'R={np.sqrt(R2)}')
print(f'MAE={MAE}')
print(f'RMSE={RMSE}')
print('R2={}'.format(R2))

score =regr_1.score(X_test, y_test)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_2))
MAE = metrics.mean_absolute_error(y_test, y_2)
R2 = r2_score(y_test, y_2)
#print(f'R={np.sqrt(R2)}')
print(f'MAE={MAE}')
print(f'RMSE={RMSE}')
print('R2={}'.format(R2))

cv = KFold(n_splits=5 ,shuffle=True,random_state=1402)

scores= CVS(regr_1,X,y,cv=cv,return_train_score=True
                   ,scoring=('r2', 'neg_mean_absolute_error','neg_mean_squared_error')
                   ,verbose=True,n_jobs=-1)


a= regr_1.feature_importances_
b=a*100
print(b)

# 画图
#plt.figure()
#plt.scatter(y_test, y_1, s=30, edgecolor="black", c="darkred", label="data")
#plt.xlabel("y_test")
#plt.ylabel("Prediction")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()