#!/usr/bin/env python
# coding: utf-8

# In[1]:


#信用违约
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

#导入数据
data = pd.read_csv('UCI_Credit_Card.csv')

#数据探索
print(data.shape)
print(data.head())
print(data.info())

#目标值分布
data['default.payment.next.month'].value_counts(normalize=True)

#数据清洗
#相关性分析
plt.figure(figsize=(25,25,))
corr = data.corr()
sns.heatmap(corr,annot=True)
#删除相关性强的特征
featues_remain = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','AGE','PAY_0','PAY_2','BILL_AMT1','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
#划分特征集和目标值
X = data[featues_remain]
y = data['default.payment.next.month']

#模型建立和训练
#分类模型
classifier = [
    SVC(random_state=1),
    DecisionTreeClassifier(random_state=1),
    RandomForestClassifier(random_state=1),
    LogisticRegression(random_state=1),
    KNeighborsClassifier(),
    AdaBoostClassifier(random_state=1)
]

#模型名称
classifier_name = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'logisticrgression',
    'kneighborsclassifier',
    'adaboostclassifier'
]

#模型参数
classifier_param_grid = [
    {'svc__C':[0.5,1,1.5],'svc__kernel':['rbf','linear','poly','sigmoid']},
    {'decisiontreeclassifier__criterion':['gini','entropy'],'decisiontreeclassifier__max_depth':[2,4,6,8],'decisiontreeclassifier__min_samples_leaf':[1,2,3,4,6,8]},
    {'randomforestclassifier__criterion':['gini','entropy'],'randomforestclassifier__max_depth':[2,4,6,8],'randomforestclassifier__n_estimators':[50,100,120,150]},
    {'logisticrgression__penalty':['l1','l2'],'logisticrgression__solver':['sag','saga','lbfgs','liblinear'],'logisticrgression__max_iter':[50,100,150]},
    {'kneighborsclassifier__n_neighbors':[2,4,6,8],'kneighborsclassifier__algorithm':['ball_tree','kd_tree','brute']},
    {'adaboostclassifier__n_estimators':[25,50,75,90,100],'adaboostclassifier__learning_rate':[0.01,0.1,0.5,1,1.5]}
]

#搭建模型
def GridSearchCV_work(pipeline,X,y,cv,param_grid,score = 'accuracy'):
    gridsearch = GridSearchCV(estimator = pipeline,param_grid = param_grid ,scoring=score,cv=cv)
    search = gridsearch.fit(X,y)
    print('最优参数',search.best_params_)
    print('最优得分%.4lf'%search.best_score_)
    
for model,model_name,model_param_grid in zip(classifier,classifier_name,classifier_param_grid):
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        (model_name,model)
    ])
    GridSearchCV_work(pipeline,X,y,5,model_param_grid,score='accuracy')

