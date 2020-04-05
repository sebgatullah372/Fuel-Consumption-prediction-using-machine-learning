# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 04:30:29 2020

@author: Arnob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

dataset = pd.read_csv('FuelConsumption.csv')
X= dataset.iloc[:,8:12]
y= dataset.iloc[:,12]
c= dataset.corr();
#Correlation value higher than 0.7, Hence Linear Regression can be applied
print("Correlation", dataset.corr())
#poly = PolynomialFeatures(degree=3)
sc= StandardScaler()
X= sc.fit_transform(X)
#X= poly.fit_transform(X)
#dataset.plot(x ='ENGINESIZE', y='CO2EMISSIONS', kind = 'scatter')
#plt.show()
#plt.scatter(dataset.ENGINESIZE, dataset.CO2EMISSIONS,  color='blue')
#plt.xlabel("Engine size")
#plt.ylabel("Emission")
#plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

reg = LinearRegression()

linear_model = reg.fit(X_train, y_train)

print(linear_model.coef_, linear_model.intercept_)
print(linear_model.score(X_train,y_train))
predictions = linear_model.predict(X_test)
y_test_val=[]
for row,value in y_test.items():
    y_test_val.append(value)
print(linear_model.score(X_test, y_test)*100)
for i in range(10):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test_val[i]))
var_score=explained_variance_score(y_test, predictions)
print("explained_variance_score", var_score*100)
