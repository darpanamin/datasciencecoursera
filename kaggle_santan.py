# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 12:05:18 2018

@author: dca
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display 
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import sys
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
import lightgbm as lgb
import lime
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

train_df = pd.read_csv('C:/Users\dcami/OneDrive/Documents/Python Scripts/train.csv',index_col='ID')
testset = pd.read_csv('C:/Users\dcami/OneDrive/Documents/Python Scripts/test.csv')


def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))

testset.pop('ID')

train_df.head()
print('Train:\t{:>6,}\t{:>6,}'.format(*train_df.shape))

traintarget = train.pop('target').mean()
train_df.std())

print(train_df.groupby('9281abeea').size())
train_df.iloc[:, 2:].nunique()
print(train_df.groupby('48df886f9').size())

target[1]
38000000

train_df_q1=train_df.iloc[:, 0:1000]
train_df_q2=train_df.iloc[:,1001:2000]
train_df_q3=train_df.iloc[:,2001:3000]
train_df_q4=train_df.iloc[:,3001:4000]
train_df_q5=train_df.iloc[:,4001:]

X25 = train_df[train_df>0].train_df.iloc[, 0:1000].nunique()
train_df[,0:1000].sum(axis=1)
target = train_df.pop('target')

max(X0)
min(X0[X0>0])

Xsum = train_df.sum(axis=1)
Xmax = train_df.max(axis=1)
Xstd = train_df.std(axis=1)
Xuniq = train_df[train_df > 0].nunique(axis=1) 
Xvals = train_df[train_df > 0].count(axis=1) 
Xmean2 =  Xsum / Xuniq
Xmean = train_df[train_df > 0].mean(axis=1) 
Xmed = train_df[train_df > 0].median(axis=1) 
Xmin = train_df[train_df > 0].min(axis=1) 
Xsumq1 = train_df_q1.sum(axis=1)
Xsumq2 = train_df_q2.sum(axis=1)
Xsumq3 = train_df_q3.sum(axis=1)
Xsumq4 = train_df_q4.sum(axis=1)
Xsumq5 = train_df_q5.sum(axis=1)
Xvalsq1 = train_df_q1[train_df_q1 > 0].count(axis=1) 
Xvalsq2 = train_df_q2[train_df_q2 > 0].count(axis=1) 
Xvalsq3 = train_df_q3[train_df_q3 > 0].count(axis=1) 
Xvalsq4 = train_df_q4[train_df_q4 > 0].count(axis=1) 
Xvalsq5 = train_df_q5[train_df_q5 > 0].count(axis=1) 

lookat = X.iloc[1,:]
lookat.to_csv('lookat.csv')
target[1]

X_Test = testset.iloc[:,1:].values

svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)

X=train_df
X_Test = 
y=target
y_log = np.log1p(y)
# Apply log transform to target variable
y_train = np.log1p(target)
y_log2 = np.log(y)

feature_selector = VarianceThreshold()
X_fs = feature_selector.fit_transform(X)
X_Test = feature_selector.transform(X_Test)

X_fs_sample_train = X_fs[0:3999,]
y_log_sample_train = y_log[0:3999,]

X_fs_sample_test = X_fs[4000:,]
y_log_sample_test = y_log[4000:,]


## SVR  

clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1) #SVR(kernel='rbf', C=1e3, gamma=0.1)
clf = clf.fit(train_df, target) 
clf = clf.fit(train_df, y_log) 
clf = clf.fit(X_fs, y_log) 


pred_results = clf.predict(X_fs)

print(np.sqrt(np.mean((pred_results-y_log) ** 2))) ## score 1.046


# split train-test
clf_samp = clf.fit(X_fs_sample_train, y_log_sample_train) 
pred_results = clf_samp.predict(X_fs_sample_test)
print(np.sqrt(np.mean((pred_results-y_log_sample_test) ** 2))) ## score 1.046




scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=100, random_state=7)

cv_results = model_selection.cross_val_score(SVC(), train_df, target, cv=kfold, scoring=scoring)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

y_pred = svr_rbf.fit(X, y_log).predict(X)

mean_squared_error(y_log, y_pred)
print(np.sqrt(np.mean((y_pred-y_log) ** 2)))




# LGBMR

FOLDS = 14
SEED = 1202
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

model = lgb.LGBMRegressor(objective='regression',
                        num_leaves=61,
                        learning_rate=0.02,
                        n_estimators=600)

predict = cross_val_predict(model, X, y_train, cv=kf)
print(np.sqrt(np.mean((predict-y_train) ** 2)))
 


predict_2 = cross_val_predict(model, X, target, cv=kf)
print(np.sqrt(np.mean((predict-y_train) ** 2)))


# MLP
X_matrix = X.as_matrix()

mlp = MLPRegressor(hidden_layer_sizes=(30,30,30))
clf = mlp.fit(X, target)
pred_results = clf.predict(X_matrix)
print(np.sqrt(np.mean((pred_results-target) ** 2)))
metrics.mean_squared_log_error(target, pred_results)

## random forrest 
NUM_OF_FEATURES = 1000

x1, x2, y1, y2 = model_selection.train_test_split(
    X, y_log, test_size=0.20, random_state=5)
model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=7)
model.fit(x1, y1)
print(rmsle(y2, model.predict(x2)))
print(np.sqrt(np.mean((model.predict(x2)-y2) ** 2)))

col = pd.DataFrame({'importance': model.feature_importances_, 'feature': X.columns}).sort_values(
    by=['importance'], ascending=[False])[:NUM_OF_FEATURES]['feature'].values

train = X[col]
test = testset[col]     
   
test.shape
  
## Submission

pred_results_test = clf.predict(X_Test)
results = np.exp(pred_results_test )

submission = pd.DataFrame()
submission['ID'] = testset['ID']
submission['target'] = results
submission.to_csv('submission.csv', index=False)
