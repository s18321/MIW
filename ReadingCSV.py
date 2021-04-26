# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:41:14 2021

Implement a random forest classifier from scratch using Gini Impurity measure. 
Classes that you should implement are "DecisionTree"  and "RandomForest".
User should be able to run program with following parameters:
-t, --train_file  : csv file with  data (required)
-s, --train_test_split : how many of datapoint will be used for tests (default and min 0.2  while max 0.8)
-c, --clasification_column : name of column in dataset with classification data (required)
--max_depth : Maximum depth of tree, dafault value is 5 and this is a ineteger
--acceptable_impurity : Level of impurity at which we no longer split nodes, default value 0
Script should output visualisation of trees as png files as well as txt file with information about correctnes level on test dataset.

As a result you should upload
code as .py
visualisation od single tree
txt file with information about prediction quality

You must use provided depression dataset and attempt to predict if someone is depressed or not.


Dataset description (columns):
Surveyid Villeid
sex
Age
Married
Numberchildren educationlevel
totalmembers (in the family) gainedasset
durableasset saveasset
livingexpenses otherexpenses
incomingsalary incomingownfarm incomingbusiness
incomingnobusiness
incomingagricultural farmexpenses
laborprimary lastinginvestment
nolastinginvestmen
depressed: [ Zero: No depressed] or [One: depressed] (Binary for target class)
the main objective is to show statistic analysis and some data mining techniques.

The dataset has 23 columns or dimensions and a total of 1432 rows or objects.

@author: s18321
"""
import pandas as pd

from keras.models import Model, Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

""" reads csv file """
depressed = pd.read_csv('b_depressed.csv')

print(depressed.info())
""" shows all unique"""
depressed.nounique()
""" drops a column """
depressed.drop(['Survey_id'], axis='columns', inplace=True)
""" show general info """
depressed.info()
""" shows how many nulls are in those columns """
depressed.isnull().sum(); 

""" describes the column where we found nulls, if large difference between min and max then we want to use that column """
depressed['no_lasting_investment'].describe()
""" change all nulls to mean of the column  """
depressed['no_lasting_investment'] = depressed['no_lasting_investment'].fillna(value=depressed['no_lasting_investment'].mean())

""" """
y =depressed.iloc[:]['depressed'].values
num_calsses = depressed['depressed'].nounique()
Y = to_categorical(y, num_classes = num_calsses)
print(Y.shape)
depressed.drop(['depressed'], axis='columns', inplace=True)

X = depressed.values
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(X_train.shape)


""" can change number of neurons to a bigger one, but it won't help much """
model = Sequential()
model.add(Dense(42, activation='relu',input_shape=[X.shape[1]]))
model.add(Dense(21, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=10, epochs=30, validation_data=(X_test, y_test))

print(history.history)
""" show plot of accuracy validation of accuracy """
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Train accuracy')
plt.legend()
plt.plot()


