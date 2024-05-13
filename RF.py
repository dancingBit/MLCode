# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:31:27 2023

@author: NJ
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold,cross_validate as CVS

from sklearn.model_selection import cross_val_score



dataset = pd.read_csv("E:\SHUIJU\senfgas 5.csv", encoding='unicode_escape')
X = dataset.iloc[:, 1:7].values 
Y = dataset.iloc[:,13].values  #423:427 #
#dataset=dataset.sample(frac=0.9)


# X = np.array(X).reshape(-1, 1)
# df = np.hstack((X,Y))

X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 1)



#Building model
rfr = RandomForestRegressor(bootstrap=False,
                           max_features=0.5,
                          min_samples_leaf=2, 
                          min_samples_split=3,
                          n_estimators=300)




# Predicting a new result
rfr.fit(X_train, y_train.ravel())
y_pred1 = rfr.predict(X_train)
y_pred2 = rfr.predict(X_test)


#train
score =rfr.score(X_train, y_train)
RMSE = np.sqrt(metrics.mean_squared_error(y_train, y_pred1))
MAE = metrics.mean_absolute_error(y_train, y_pred1)
R2= r2_score(y_train, y_pred1)
#print(f'R={np.sqrt(R2)}')

print('R2={}'.format(R2))
print(f'MAE={MAE}')
print(f'RMSE={RMSE}')
#test
score =rfr.score(X_test, y_test)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
MAE = metrics.mean_absolute_error(y_test, y_pred2)
R2 = r2_score(y_test, y_pred2)
#print(f'R={np.sqrt(R2)}')
print('R2={}'.format(R2))
print(f'MAE={MAE}')
print(f'RMSE={RMSE}')

cv = KFold(n_splits=5 ,shuffle=True,random_state=1402)

scores= CVS(rfr,X,Y,cv=cv,return_train_score=True
                   ,scoring=('r2','neg_mean_absolute_error', 'neg_mean_squared_error')
                   ,verbose=True,n_jobs=-1)


a= rfr.feature_importances_
b=a*100
print(b)


