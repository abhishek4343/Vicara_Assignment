# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:27:39 2021

@author: 19abh
"""

#Getting libraries
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('bmh')
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import time

from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

#Now let us read all data files we have and perform EDA on the dataset
Train = pd.DataFrame()

path = r'C:\Users\19abh\Desktop\DummyProjects\vicara\Data' # use your path
all_files = glob.glob(path + "/*.csv")

# Appending all individual data as Train data
for filename in all_files:
    temp = pd.read_csv(filename, index_col=None, header=None)
    Train = Train.append(temp)

# Renaming columns based on their understanding
Train.columns = ['sequential_number', 'x_acceleration', 'y_acceleration', 'z_acceleration', 'label']
Train.head(5)

#Let us get some basic information on the Train data and also check 'label' dependent vaiable for any outlier
Train.info()
Train['label'].value_counts()

#Here, following are the meaning of label values based on README file provided in Data:
#
#--- 1: Working at Computer
#--- 2: Standing Up, Walking and Going up\down stairs
#--- 3: Standing
#--- 4: Walking
#--- 5: Going Up\Down Stairs
#--- 6: Walking and Talking with Someone
#--- 7: Talking while Standing
#
#Here we notice there are 3719 occurences of '0' in our dependent variable. So let us remove the outlier from Train data.
Train = Train[Train['label'] != 0]

#Let us do some EDA on our data. First let us check the distribution of various labels across our data.
print(Train['label'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(Train['label'], color='g', bins=100, hist_kws={'alpha': 0.4});

#As we can see we have majority of labels for individuals that are 'Working at Computer' followed by 'Talking while Standing', 'Walking' and 'Standing'.
#Let us see the distribution of our predictor features.
Train.iloc[:,0:4].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

#We can see 'x_acceleration' and 'z_acceleration' have majority of data points on value 2000 while that of 'y_acceleration' is between 2000-3000.
#'sequential_number' however seems very off so we will be dropping this feature for now.
del Train['sequential_number']

#Let us also check correlation of independent variables to dependent variable
Train_corr = Train.corr()['label'][:-1] # -1 because the latest row is SalePrice
features_list = Train_corr[abs(Train_corr) > 0].sort_values(ascending=False)
print("The {} independent variables' correlated values with label are:\n{}".format(len(features_list), features_list))

#As we can see they have really less correlation with dependent variable 'label'. So no need of removing any feature now. Let us normalize the data.
min_max_scaler = preprocessing.MinMaxScaler()

Train.iloc[:,0:3] = Train.iloc[:,0:3].astype(float)
Train.iloc[:,0:3] = min_max_scaler.fit_transform(Train.iloc[:,0:3].values)

#Splitting original Train data into Train/ Test set
X = Train.iloc[:,0:3]
y = Train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Method one:
#Running a base model. We can choose any model however here I am selecting RandomForestClassifier for classification and then calculate it's accuracy.
clf = RandomForestClassifier()

start_time = time.time()
clf.fit(X_train, y_train)
print('Time taken to fit is: ', time.time() - start_time, ' seconds')
# output -
# Time taken to fit is:  34.92870855331421  seconds

y_pred = clf.predict(X_test)
print('Accuracy is: ', accuracy_score(y_test, y_pred)*100, ' percent')
# output - 
# Accuracy is:  71.0556543853374  percent

#Method Two:
#Selecting best base model and improving it further using RandomizedSearchCV and GridSearchCV.
#I am considering RidgeClassifier, SGDClassifier and RandomForestClassifier as example here, we can add more classifiers too.
list_of_models = [RidgeClassifier(), SGDClassifier(), RandomForestClassifier()]
list_of_acc = []

for i in list_of_models:
    start_time = time.time()
    i.fit(X_train, y_train)
    print(time.time() - start_time)

    predicted = i.predict(X_test)
        
    cc = accuracy_score(y_test, predicted)
    list_of_acc.append(cc)
    print('##################')

a = max([(v,i) for i,v in enumerate(list_of_acc)])
b = a[1]
c = list_of_models[b]
print('Best Model is: ', c)
# output - 
# Best Model is:  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)

# Let us get few parameters of RandomForestClassifier for RandomizedSearchCV
gridRandomForestClassifier = {'n_estimators': [100,1000],
               'criterion': ['gini', 'entropy'],
               'min_samples_leaf': [1,2,5],
               'max_leaf_nodes' : [100,1000]}

# Using RandomizedSearchCV
op = RandomizedSearchCV(c, gridRandomForestClassifier, cv=3, random_state=42, n_iter=5)

start_time = time.time()
op.fit(X_train, y_train) 
print(time.time() - start_time)
# output -
# 7036.775358200073

predicted = op.predict(X_test)
print('Accuracy is: ', accuracy_score(y_test, predicted)*100, ' percent')
# output -
# Accuracy is:  74.3024884621263  percent

print('Best predictor after RandomizedSearchCV is: ', op.best_estimator_)
# output -
# Best predictor after RandomizedSearchCV is:  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=1000,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=5, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)

# Using GridSearchCV
opg =  GridSearchCV(c, gridRandomForestClassifier, cv=3)

start_time = time.time()
opg.fit(X_train, y_train) 
print(time.time() - start_time)

predicted = opg.predict(X_test)
print('Accuracy is: ', accuracy_score(y_test, predicted)*100, ' percent')

print('Best predictor after GridSearchCV is: ', opg.best_estimator_)

# Saving the predictions
Output = pd.DataFrame()
Output['predicted'] = predicted

Output.to_csv('Output.csv')

