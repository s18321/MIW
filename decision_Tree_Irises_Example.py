# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:25:37 2021

@author: s18321
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, export_graphviz

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)
print(type(x))
print(dataset.feature_names)
print(dataset.target_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(x_train, y_train)

export_graphviz(classifier, 'classifier.dot', feature_names=dataset.feature_names, class_names=dataset.target_names)

! dot -Tpng classifier.dot -o classifier.png

img = cv2.imread('classifier.png')
plt.figure(figsize=(13,13))
ax = plt.gca()
ax.axes.get_xaxis().set_visable(False)
ax.axes.get_yaxis().set_visable(False)
ax.imshow(img)
plt.show()

print(dataset.target_names[classifier.predict(x_test[3:12,:]))
print(dataset.target_names[y_test[3:12]])

classifier = RandomForestClassifier(max_depth=3, n_estimators=9)
classifier.fit(x_train, y_train)
print(len(classifier.estimators_))

import os

for tree_id, tree in enumerate(classifier.estimators_)
export_graphviz(tree, f'tree{tree_id:02d}.dot', )
                           


