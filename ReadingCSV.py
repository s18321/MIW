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

You must use provided depression dataset and attempt to predict if someone is dataSet or not.


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
dataSet: [ Zero: No dataSet] or [One: dataSet] (Binary for target class)
the main objective is to show statistic analysis and some data mining techniques.

The dataset has 23 columns or dimensions and a total of 1432 rows or objects.

@author: s18321
"""
import pandas as pd
import os

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()


# Data validation for test_split
def test_split(split_num):
    if 0.8 <= split_num or 0.2 >= split_num:
        raise argparse.ArgumentTypeError
    return split_num


parser.add_argument('-t', '--train_file', help="csv file with  data", default='b_depressed.csv', required=True)
parser.add_argument('-s', '--train_test_split', help="how many of datapoint will be used for tests", type=test_split,
                    default=0.2, required=False)
parser.add_argument('-c', '--clasification_column', help="name of column in dataset with classification data",
                    required=True)
parser.add_argument('--max_depth', help="Maximum depth of tree", type=int, default=5, required=False)
parser.add_argument('--acceptable_impurity', help="Level of impurity at which we no longer split nodes", type=float,
                    default=0, required=False)

""" prints all passed arguments """
args = parser.parse_args()
print("train_file: ", args.train_file)
print("train_test_split: ", args.train_test_split)
print("clasification_column: ", args.clasification_column)
print("max_depth: ", args.max_depth)
print("acceptable_impurity: ", args.acceptable_impurity)

""" reads csv file """
dataSet = pd.read_csv(args.train_file)
""" drops all rows with null values """
dataSet = dataSet.dropna()

y = dataSet.iloc[:][args.clasification_column].values

target_names = args.clasification_column

""" checks how many values can classification column have and sets it as categories """
# num_classes = dataSet[args.clasification_column].nounique()
Y = pd.Categorical(y)
print(f'Shape of y {Y.shape}')
dataSet.drop([args.clasification_column], axis='columns', inplace=True)

X = dataSet.values

feature_names = dataSet.columns
print(f'Shape of data {X.shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.train_test_split)

tree_classifier = DecisionTreeClassifier(max_depth=args.max_depth, min_impurity_split=args.acceptable_impurity)
tree_classifier.fit(X_train, y_train)  # add tree model

# export_graphviz(classifier, 'classifier.dot', feature_names=dataset.feature_names, class_names=dataset.target_names)

tree_score = tree_classifier.score(X_test, y_test)
print(f'Score of tree {tree_score}')

forest_classifier = RandomForestClassifier(max_depth=args.max_depth, n_estimators=9,
                                           min_impurity_split=args.acceptable_impurity)
forest_classifier.fit(X_train, y_train)

forest_score = forest_classifier.score(X_test, y_test)
print(f'Score of forest  {forest_score}')

print(len(forest_classifier.estimators_))
print(forest_classifier.estimators_)
# Saving a tree visualisation to a png file
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(tree_classifier,
                   feature_names=feature_names,
                   class_names=target_names,
                   filled=True)
fig.savefig("decision_tree_visualized.png")

# Saving scoring to a txt file
score_file = open("scoring_of_classifiers.txt", "w")
L = ['Scores achieved and arguments passed \n',
     'Arguments passed: \n',
     f'train_file: {args.train_file} \n',
     f'train_test_split: {args.train_test_split} \n',
     f'clasification_column: {args.clasification_column} \n',
     f'max_depth: {args.max_depth} \n',
     f'acceptable_impurity: {args.acceptable_impurity} \n',
     'Scores: \n'
     f'Score of RandomForestClassifier: {forest_score} \n',
     f'Score of DecisionTreeClassifier: {tree_score} \n']
score_file.writelines(L)
score_file.close()  # to change file access modes

# (clf,
#                  feature_names=iris.feature_names,
#                 class_names=iris.target_names,
#                filled=True)
# TO DO: get score to external txt file and get graphiz to make a png

# DOT data
# dot_data = tree.export_graphviz(tree_classifier, out_file=None,
#                                 feature_names=feature_names,
#                                 class_names=target_names,
#                                 filled=True)
#
# # Draw graph
# graph = graphviz.Source(dot_data, format="png")
# print(graph)
#
# graph.render("decision_tree_graphivz")
# 'decision_tree_graphivz.png'

# for tree_id, tree in enumerate(classifier.estimators_):
#     export_graphviz(tree, f'tree{tree_id:02d}.dot', )
