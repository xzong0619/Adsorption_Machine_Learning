# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 20:55:44 2021

@author: zongx
"""
# Import necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_validate, LeaveOneOut, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pprint import pprint

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

# Set up cross-validation scheme                                  
kf = KFold(n_splits = 10, shuffle = True, random_state = 0)

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

#%% Tune model hyperparameter: n_estimators (100)
# Create a based model
model = ExtraTreesRegressor(random_state = 0)

# Instantiate the grid search model
gsc = GridSearchCV(estimator = model, 
                           param_grid = {'n_estimators': range(1, 101, 1)},
                           scoring = 'neg_mean_squared_error',
                           cv = 10, n_jobs = -1, verbose = 2,
                           return_train_score = True)

# Fit the grid search to the data
etr = gsc.fit(X_train, Y_train.ravel())
grid_result = pd.DataFrame(data = etr.cv_results_)
print("Best: %f using %s" % (etr.best_score_, etr.best_params_))

test_score = grid_result['mean_test_score']
test_std = grid_result['std_test_score']

train_score = grid_result['mean_train_score']
train_std = grid_result['std_train_score']

# Visualize the learning curve
x = np.linspace(1, 100, num = 100)
fig, ax = plt.subplots()

ax.plot(x, abs(train_score), '-o',  color = 'b',  markerfacecolor = "None", label = 'Train score')
ax.plot(x, abs(test_score), '-o', color = 'g', markerfacecolor = "None", label = 'Validation score')
ax.set_xlabel('Number of Estimators')
ax.set_ylabel('Scores: MSE')
ax.legend()

#%% Tune model hyperparameter: max_features (6)
# Create a based model
model = ExtraTreesRegressor(random_state = 0)

# Instantiate the grid search model
gsc = GridSearchCV(estimator = model, 
                           param_grid = {'max_features': range(1, 11, 1)},
                           scoring = 'neg_mean_squared_error',
                           cv = 10, n_jobs = -1, verbose = 2,
                           return_train_score = True)

# Fit the grid search to the data
etr = gsc.fit(X_train, Y_train.ravel())
grid_result = pd.DataFrame(data = etr.cv_results_)
print("Best: %f using %s" % (etr.best_score_, etr.best_params_))

test_score = grid_result['mean_test_score']
test_std = grid_result['std_test_score']

train_score = grid_result['mean_train_score']
train_std = grid_result['std_train_score']

# Visualize the learning curve
x = np.linspace(1, 10, num = 10)
fig, ax = plt.subplots()

ax.plot(x, abs(train_score), '-o',  color = 'b',  markerfacecolor = "None", label = 'Train score')
ax.plot(x, abs(test_score), '-o', color = 'g', markerfacecolor = "None", label = 'Validation score')
ax.set_xlabel('Maximum Features')
ax.set_ylabel('Scores: MSE')
ax.legend()

#%% Tune model hyperparameter: min_samples_split
# Create a based model
model = ExtraTreesRegressor(random_state = 0)

# Instantiate the grid search model
gsc = GridSearchCV(estimator = model, 
                           param_grid = {'min_samples_split': range(2, 11, 1)},
                           scoring = 'neg_mean_squared_error',
                           cv = 10, n_jobs = -1, verbose = 2,
                           return_train_score = True)

# Fit the grid search to the data
etr = gsc.fit(X_train, Y_train.ravel())
grid_result = pd.DataFrame(data = etr.cv_results_)
print("Best: %f using %s" % (etr.best_score_, etr.best_params_))

test_score = grid_result['mean_test_score']
test_std = grid_result['std_test_score']

train_score = grid_result['mean_train_score']
train_std = grid_result['std_train_score']

# Visualize the learning curve
x = np.linspace(2, 10, num = 9)
fig, ax = plt.subplots()

ax.plot(x, abs(train_score), '-o',  color = 'b',  markerfacecolor = "None", label = 'Train score')
ax.plot(x, abs(test_score), '-o', color = 'g', markerfacecolor = "None", label = 'Validation score')
ax.set_xlabel('Min_samples_split')
ax.set_ylabel('Scores: MSE')
ax.legend()


#%% Fit the model using the optimal parameters
etr_opt = ExtraTreesRegressor(n_estimators = 100, #92
                              max_features = 9, #9
                              random_state = 0)

# Cross validation
opt_train_error, opt_cv_error = cross_validation(X_train, Y_train.ravel(),
                                                 model = etr_opt,
                                                 cv_method = kf)

# Calculate the error on test data
etr_opt.fit(X_train, Y_train.ravel())
opt_pred = etr_opt.predict(X_test)
opt_test_error= np.sqrt(mean_squared_error(Y_test, opt_pred))

# Print out model performance metrics
print(' Training error of {0:5.2f}'.format(opt_train_error))
print(' Validation error of {0:5.2f}'.format(opt_cv_error))
print(' Test error of {0:5.2f}'.format(opt_test_error))

#%% Parity plot to visualize prediction results
opt_train_pred = etr_opt.predict(X_train)
plt.scatter(y_train, opt_train_pred)
plt.scatter(y_test, opt_pred)

plt.xlabel('Binding Energy from DFT (eV)')
plt.ylabel('Binding Energy from ETR (eV)')
plt.title('ML vs DFT Energy for ETR Model')

#%% Explore feature importance
# Get numerical feature importances
importances = list(etr_opt.feature_importances_)

# Visualizations: Bar plot for feature importances
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