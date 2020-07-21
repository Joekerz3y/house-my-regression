# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:21:16 2020

@author: David
"""

import pandas as pd 
import random
import xgboost as xgb 

from sklearn.metrics import mean_squared_error

import numpy as np


from sklearn.preprocessing import OneHotEncoder

random.seed(42)


#read in modelling data

df = pd.read_csv("train.csv")

df_test = pd.read_csv("test.csv")


#drop ID column 

df = df[df.columns.drop("Id")]

df_test_id = df_test[["Id"]]

df_test = df_test[df_test.columns.drop("Id")]


#separate target variable 

X, y = df.iloc[:,:-1],df.iloc[:,-1]


#names

df_str = X.select_dtypes(['object'])

names = df_str.columns

#one hot encode 

X_train_processed = pd.get_dummies(X, prefix_sep="__",
                              columns=names)


X_test_processed = pd.get_dummies(df_test, prefix_sep="__",
                              columns=names)



data_dmatrix = xgb.DMatrix(data=X_train_processed,label=y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_processed, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 1, n_estimators = 100)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
