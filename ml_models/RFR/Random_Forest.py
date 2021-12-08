# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:39:04 2021

@author: zongx
"""
# Import necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_validate, LeaveOneOut, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score

#%% Data Preprocessing
# Load the dataset
data = pd.read_excel('Dataset.xlsx')
# Return column names
labels = list(data.columns)
# column_names = ['CN', 'n_metal', 'GCN', 'Valency', 'chi', 'Bader', 'ed', 'wd', 'E_bind', 'E_ads', 'n_H', 'n_H_NN']

descriptors = ['CN', 'n_metal', 'GCN', 'Valency', 'chi', 'Bader',
               'ed', 'wd', 'n_H', 'n_H_NN']

# Select data for input and output
X = data.iloc[:, [2, 4, 5, 6, 7, 8, 9, 10, 13, 14]]

# Choose binding energy as response for now
# Two responses in total: Binding energy or adsorption energy
Y = data.iloc[:, [11]]

# Standardize the data to a unit variance and 0 mean
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_std = scaler.scale_
X_mean = scaler.mean_

# Check if there have a unit variance and zero mean
Xs_std = np.std(X_scaled, axis = 0)
print('The std of each column in standardized X:')
print(Xs_std)
Xs_mean = np.mean(X_scaled, axis = 0)
print('The mean of each column standardized X:')
print(Xs_mean)

# Split data into training and test set, set up cross-validation
# Specify random_state to make results reproducible
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

# Convert dataframe to numpy array
Y_train = y_train.to_numpy()
Y_test = y_test.to_numpy()

# check train and test set sizes
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Set up cross-validation scheme                                  
kf = KFold(n_splits = 10, shuffle = True, random_state = 0)

#%% Create helper functions
def cross_validation(X, y, model, cv_method): 
    """Cross-validation
    :param X: feature matrix
    :param y: response
    :type y: 1D np array
    :param model: regression model
    :type model: sklearn object
    :param cv_method: cross validation method
    :type cv_method: cv object
    :return: mean cv error  
    :rtype: float
    """    

    scores  = cross_validate(model, X, y, cv = cv_method,
                                scoring=('neg_mean_squared_error'),
                                return_train_score=True)
    
    # Use RMSE as the loss function (score)
    # Export train RMSE
    train_RMSE = np.sqrt(np.abs(scores['train_score']))  
    train_RMSE_mean = np.mean(train_RMSE)
    
    # Export cv RMSE (test set in cv is the validation set)
    cv_RMSE = np.sqrt(np.abs(scores['test_score']))  
    cv_RMSE_mean = np.mean(cv_RMSE)

    
    return train_RMSE_mean, cv_RMSE_mean

#%% Hyperparameter Tuning: GridSearch
# Create the parameter grid
param_grid = {'n_estimators': range(1, 101, 1),
              #'max_features': range(1, 11, 1)
              }

# Create a based model
rf = RandomForestRegressor(random_state = 0)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           scoring = 'neg_mean_squared_error',
                           cv = 10, n_jobs = -1, verbose = 2,
                           return_train_score = True)

# Fit the grid search to the data
rfr = grid_search.fit(X_train, Y_train.ravel())
print(rfr.best_params_)
grid_result = pd.DataFrame(data = rfr.cv_results_)

test_score = grid_result['mean_test_score']
test_std = grid_result['std_test_score']

train_score = grid_result['mean_train_score']
train_std = grid_result['std_train_score']

#%% Visualize the learning curve for n_estimators
x = range(1, 101, 1)
fig, ax = plt.subplots(figsize = (5, 5))

ax.plot(x, abs(train_score), '-o',  color = 'b',  markerfacecolor = "None", label = 'Train score')
ax.plot(x, abs(test_score), '-o', color = 'g', markerfacecolor = "None", label = 'Validation score')
ax.set_xlabel('N_estimators')
ax.set_ylabel('Scores')
ax.legend()

#%% Fit the model using the optimal hyperparameters
rf_opt = RandomForestRegressor(n_estimators = 55, max_features = 7, random_state=0)
rf_opt.fit(X_train, Y_train.ravel())

# Cross validation
opt_train_error, opt_cv_error = cross_validation(X_train, Y_train.ravel(),
                                                 model = rf_opt,
                                                 cv_method = kf)

# Calculate the error on test data
y_pred_train = rf_opt.predict(X_train)
y_predict_test = rf_opt.predict(X_test)
opt_test_error = np.sqrt(mean_squared_error(y_test, y_predict_test))

# Print out model performance metrics
print(' Training error of {0:5.2f}'.format(opt_train_error))
print(' Validation error of {0:5.2f}'.format(opt_cv_error))
print(' Test error of {0:5.2f}'.format(opt_test_error))

#%% Parity plot to visualize prediction results
opt_train_pred = rf_opt.predict(X_train)
plt.scatter(y_train, opt_train_pred)
plt.scatter(y_test, y_predict_test)

plt.xlabel('Binding Energy from DFT (eV)')
plt.ylabel('Binding Energy from KRR (eV)')
plt.title('ML vs DFT Energy for RFR Model')

#%% Explore feature importance
# Get numerical feature importances
importances = list(rf_opt.feature_importances_)

# List of tuples with viriable and importance
feature_importances = [(des, round(importance, 4)) for des, importance in zip(descriptors, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:10} Importance: {}'.format(*pair)) for pair in feature_importances]

#%% Visualizations: Bar plot for feature importances
# List of x locations for plotting
x_values = list(range(len(descriptors)))
# Make a bar chart
plt.bar(x_values, importances)
# Tick label for x axis
plt.xticks(x_values, descriptors, rotation = 30)
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Features')
plt.title('Feature Importances')

#%% New random forest with only the two most important variables and the same number of estimators
"""
rf_most_important = RandomForestRegressor(n_estimators = n_estimators_opt, random_state=0)
important_indices = [descriptors.index('Valency'), descriptors.index('Bader')]
train_important = X_train[:, important_indices]
test_important = X_test[:, important_indices]

# Train the random forest model
rf_most_important.fit(train_important, Y_train.ravel())

# Make predictions and determine the error
y_predict_important = rf_most_important.predict(test_important)
RMSE_important = np.sqrt(mean_squared_error(Y_test, y_predict_important))
print('RMSE of test set using only two important features is: {}'.format(RMSE_important))
# RMSE of test set using only two important features is: 0.4904838801721426

"""
