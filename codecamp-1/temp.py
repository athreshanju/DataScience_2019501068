# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.stats import skew
from scipy.stats import uniform
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mserr

path = "C:\\Users\\Chowdary\\Desktop\\datasets\\"
train = path+"train.csv"
test = path+"test.csv"
trainx_df = pd.read_csv(train,index_col='Id')
trainy_df = trainx_df['SalePrice']
trainx_df.drop('SalePrice',axis=1,inplace=True)
testx_df = pd.read_csv(test,index_col='Id')
sample_size = len(trainx_df)
columns_with_null_values=[]
columns_with_null_values=[[col,float(trainx_df[col].isnull().sum())/float(sample_size)]
                          for col in trainx_df.columns if
                          trainx_df[col].isnull().sum()]
columns_to_drop = [x for (x,y) in columns_with_null_values if y>.3]
trainx_df.drop(columns_to_drop,axis=1,inplace=True)
testx_df.drop(columns_to_drop,axis=1,inplace=True)
categorical_columns=[col for col in trainx_df.columns if 
                     trainx_df[col].dtype==object]
#categorical_columns.append('MSSubClass')
ordinal_columns = [col for col in trainx_df.columns if col not in categorical_columns]
dummy_row=list()
for col in trainx_df.columns:
    if col in categorical_columns:
        dummy_row.append("dummy")
    else:
        dummy_row.append("")
#print(dummy_row)

new_row=pd.DataFrame([dummy_row],columns=trainx_df.columns)
trainx_df=pd.concat([trainx_df,new_row],axis = 0,ignore_index=True)
testx_df=pd.concat([testx_df],axis=0,ignore_index=True)
for col in categorical_columns:
    trainx_df[col].fillna(value="dummy",inplace=True)
    testx_df[col].fillna(value="dummy",inplace=True)
enc = OneHotEncoder(drop='first',sparse=False)
enc.fit(trainx_df[categorical_columns])
#print(enc.get_feature_names(categorical_columns))
trainx_enc=pd.DataFrame(enc.transform(trainx_df[categorical_columns]))
testx_enc = pd.DataFrame(enc.transform(testx_df[categorical_columns]))
trainx_enc.columns=enc.get_feature_names(categorical_columns)
testx_enc.columns=enc.get_feature_names(categorical_columns)
trainx_df=pd.concat([trainx_df[ordinal_columns],trainx_enc],axis=1,ignore_index=
                    True)
testx_df=pd.concat([testx_df[ordinal_columns],testx_enc],axis=1,ignore_index=
                    True)
trainx_df.drop(trainx_df.tail(1).index,inplace=True)
imputer = KNNImputer(n_neighbors=2)
imputer.fit(trainx_df)
trainx_df_filled=imputer.transform(trainx_df)
trainx_df_filled = pd.DataFrame(trainx_df_filled,columns=trainx_df.columns)
testx_df_filled=imputer.transform(testx_df)
testx_df_filled = pd.DataFrame(testx_df_filled,columns=testx_df.columns)
testx_df_filled.reset_index(drop=True,inplace=True)
print(trainx_df_filled.isnull().sum())
scaler = preprocessing.StandardScaler().fit(trainx_df)
trainx_df=scaler.transform(trainx_df)
testx_df=scaler.transform(testx_df)
print(trainx_df,testx_df)
X_train, X_test, y_train, y_test = train_test_split(trainx_df_filled, trainy_df.values.ravel(), test_size = 0.3, random_state = 42)
print(X_train,X_test,y_train,y_test)
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))
score_train=[]
score_test=[]
mse_train=[]
mse_test=[]
alpha=[]
for sigma in np.linspace(10.05, 71.05, 10):
    alpha.append(sigma)
    reg = Ridge(alpha = sigma, tol = 0.0001)
    reg = reg.fit(X_train, y_train)
    pred = pd.DataFrame(reg.predict(testx_df_filled))
    score_train.append(round(reg.score(X_train, y_train),10))
    score_test.append(round(reg.score(X_test, y_test),10))
    mse_train.append(round(mserr(y_train,reg.predict(X_train)),4))
    mse_test.append(round(mserr(y_test,reg.predict(X_test)),4))
    testpred = pd.DataFrame(reg.predict(testx_df_filled))
    testpred.to_csv("test_pred.csv")
print(alpha, '\n', score_train, '\n', score_test)
plt.figure(1)
plt.plot(alpha, score_train, 'g--', label = "train_score")
plt.plot(alpha, score_test, 'r-o', label = "test_score")
plt.xlabel = 'Alpha'
plt.legend()
plt.figure(2)
plt.plot(alpha, mse_train, 'y--',label = "train_mse")
plt.plot(alpha, mse_test, 'c-o',label = "test_mse")
plt.xlabel = 'Alpha'
plt.legend()
plt.show()








    






