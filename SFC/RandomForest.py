# -*- coding: utf-8 -*-
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier # random forest
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix,classification_report # matrics
import gzip


def main(ifname=None, delimiter=None, columns=None):
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    # print(train_data.head())
    # print(train_data.shape)
    
    # 数量统计
    # category = ['Category', 'DayOfWeek', 'PdDistrict']
    # for i in category:
    #     train_data[i].value_counts().plot.bar()
    #     plt.savefig('fig/%s' % i)
    #     plt.show()
    
    # 日期与犯罪类型
    # gp = train_data.groupby(by=['DayOfWeek','Category'])
    # gp = gp.size().reset_index(name='times')
    # print(gp.sort_values(['DayOfWeek', 'Category', 'times'], ascending=True).groupby('DayOfWeek').head(3))


    #  Data Preprocessing
    train_data = DatePreprocessing(train_data)
    train_data = train_data.drop('Resolution', axis=1)
    
    Cat = LabelEncoder()
    Cat.fit(train_data['Category'])
    train_data['Cat'] = Cat.transform(train_data['Category'])
    
    train_data = train_data.drop('Category', axis=1)
    train_data = train_data.drop('Descript', axis=1)
    
    test_data = DatePreprocessing(test_data)
    
    # print(train_data.head())
    
    # Prepare training
    features = ['PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE',
       'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK',
       'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL',
       'PdDistrict_TENDERLOIN', 'DayOfWeek_Friday', 'DayOfWeek_Monday',
       'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday',
       'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'Month', 'Day', 'Hour']
    
    target = 'Cat'
    
    X = train_data[features]
    Y = train_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.2, 
                                                    random_state=100, 
                                                    stratify=Y)
    scaler=preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # pipeline = make_pipeline(preprocessing.StandardScaler(), 
    #                   SVC(class_weight='balanced', gamma='auto', probability=True))
    pipeline = make_pipeline(preprocessing.StandardScaler(), 
                          RandomForestClassifier(n_estimators=50, min_samples_split=100))
                      
    # kernel: rbf, poly, sigmoid  C: 0.5-1.5
    # hyperparameters= {'svc__kernel': ['poly'],
    # 'svc__C': [0.7]}
    
    # max_features: auto, sqrt, log2
    # max_depth: 3-8
    hyperparameters= {'randomforestclassifier__max_features': ['log2'],
    'randomforestclassifier__max_depth': [6]}
    
    # cross-validation
    clf = GridSearchCV(pipeline, hyperparameters, cv=3)
    clg = clf.fit(X_train, y_train)
    print(clg.best_params_)

    # prediction result
    y_pred = clf.predict_proba(X_test)
    
    # print(r2_score(y_test, y_pred))
    # print(mean_squared_error(y_test, y_pred))

    
    submission = pd.DataFrame(y_pred,columns=Cat.classes_)
    # submission['Id'] = X_test.Id.tolist()
    print(submission)
# 
#     #submission_cols = [test.columns[0]]+list(test.columns[14:])
#     submission.to_csv(gzip.open('first_run.csv.gz','wt'), index = False)

    
    
def DatePreprocessing(data):
    # dummies for District and Day of week
    data = pd.get_dummies(data, columns=['PdDistrict'])
    data = pd.get_dummies(data, columns=['DayOfWeek'])
    
    # processing Dates
    data["Month"] = data["Dates"].map(lambda x: int(x[5:7]))
    data["Day"] = data["Dates"].map(lambda x:int(x[8:10]))
    data["Hour"] = data["Dates"].map(lambda x:int(x[11:13]))
    
    # drop useless features
    data = data.drop('Dates', axis=1)
    data = data.drop('Address', axis=1)
    data = data.drop('X', axis=1)
    data = data.drop('Y', axis=1)
    
    
    return data

    
    
    
if __name__ == '__main__':
    main()
