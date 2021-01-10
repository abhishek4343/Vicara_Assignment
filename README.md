# Vicara_Assignment

The dataset collects data from a wearable accelerometer mounted on the chest.
I have used 3 DoF sensors as predictors to classify 7 labels that we had in our 15 different datasets. Outliers were removed.
Basic EDA was done to understand distribution of data and then data was normalized before it was fed to model.
Basic model initially used was 'RandomForestClassifier'. Since we had a very large dataset, Ramdom Forest was used as it maintains the accuracy of large proportion of data and result is stable.
Have also added a lit of classfiers with default parameters and witten a simple code which selects the best base model on basis of 'Accuracy'.
Then we further improve the results using 'RandomizedSearchCV' and 'GridSearchCV'.
Have added various comments in 'Assignment.py' explaining steps as well.

# Results (Accuracy):

1. Using default RandomForestClassifier - 71.06%
2. Using RandomizedSearchCV on RandomForestClassifier - 74.30%
3. Using GridSearchCV on RandomForestClassifier - ** Unable to fetch result due to excess memory consumption which restarted Python on my laptop **
