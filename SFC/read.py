# -*- coding: utf-8 -*-
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def main(ifname=None, delimiter=None, columns=None):
    train_data = pd.read_csv('data/train.csv')
    # print(train_data.head())
    # print(train_data.shape)
    
    # 数量统计
    category = ['Category', 'DayOfWeek', 'PdDistrict']
    for i in category:
        train_data[i].value_counts().plot.bar()
        plt.savefig('fig/%s' % i)
        plt.show()
    
    # 日期与犯罪类型
    gp = train_data.groupby(by=['DayOfWeek','Category'])
    gp = gp.size().reset_index(name='times')
    print(gp.sort_values(['DayOfWeek', 'Category', 'times'], ascending=True).groupby('DayOfWeek').head(3))

    
    

    
if __name__ == '__main__':
    main()