# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

USAinsurance = pd.read_csv('insurance.csv')

USAinsurance[['sex', 'smoker', 'region']] = USAinsurance[['sex', 'smoker', 'region']].astype('category')
USAinsurance.dtypes

##Converting category labels into numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(USAinsurance.sex.drop_duplicates())
USAinsurance.sex = label.transform(USAinsurance.sex)
label.fit(USAinsurance.smoker.drop_duplicates())
USAinsurance.smoker = label.transform(USAinsurance.smoker)
label.fit(USAinsurance.region.drop_duplicates())
USAinsurance.region = label.transform(USAinsurance.region)
USAinsurance.dtypes



X = USAinsurance [['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = USAinsurance ['charges']

X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

pickle.dump(lm, open('model.pickle', 'wb'))