"""
Author: s18321 Marcin Rybiński
"""
import pandas as pd

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
                    default='depressed',
                    required=True)
parser.add_argument('--max_depth', help="Maximum depth of tree", type=int, default=5, required=False)
parser.add_argument('--acceptable_impurity', help="Level of impurity at which we no longer split nodes", type=float,
                    default=55, required=False)


args = parser.parse_args()


""" reads csv file """
dataSet = pd.read_csv(args.train_file)
""" drops all rows with null values and data from first column which is usually id data (not useful for data mining)"""
dataSet = dataSet.dropna()
dataSet.drop([dataSet.columns[0]], axis='columns', inplace=True)

# dataset preparation
y = dataSet.iloc[:][args.clasification_column].values
target_names = args.clasification_column
Y = pd.Categorical(y)
dataSet.drop([args.clasification_column], axis='columns', inplace=True)
X = dataSet.values
feature_names = dataSet.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.train_test_split)

# Using DecisionTreeClassifier, training and testing
tree_classifier = DecisionTreeClassifier(max_depth=args.max_depth, min_impurity_split=args.acceptable_impurity)
tree_classifier.fit(X_train, y_train)
tree_score = tree_classifier.score(X_test, y_test)
# Using RandomForestClassifier, training and testing
forest_classifier = RandomForestClassifier(max_depth=args.max_depth, n_estimators=9,
                                           min_impurity_split=args.acceptable_impurity)
forest_classifier.fit(X_train, y_train)
forest_score = forest_classifier.score(X_test, y_test)

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
     'Scores (prediction quality): \n'
     f'Score of RandomForestClassifier: {forest_score} \n',
     f'Score of DecisionTreeClassifier: {tree_score} \n']
score_file.writelines(L)
score_file.close()
