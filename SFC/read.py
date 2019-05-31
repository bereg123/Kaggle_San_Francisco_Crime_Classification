# -*- coding: utf-8 -*-
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor # random forest
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix,classification_report # matrics


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
    test_data = DatePreprocessing(test_data)
    

    
    
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
