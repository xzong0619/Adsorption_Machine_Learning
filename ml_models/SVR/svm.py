# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:23:23 2021

@author: zongx
"""
# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV

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

#%% Hyperparameters Tuning using GridSearch
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9],
              'kernel': ['rbf', 'poly']}

svr = GridSearchCV(estimator = SVR(), 
                   param_grid = param_grid, 
                   scoring = 'neg_mean_squared_error',
                   cv = 10,
                   return_train_score = True)

model = svr.fit(X_train, Y_train.ravel())
grid_result = pd.DataFrame(data=model.cv_results_)
print("Best: %f using %s" % (model.best_score_, model.best_params_))

test_score = grid_result['mean_test_score']
test_std = grid_result['std_test_score']

train_score = grid_result['mean_train_score']
train_std = grid_result['std_train_score']

#%% GridSearch: C
svr_c = GridSearchCV(estimator = SVR(gamma = 0.01),
                     param_grid = {'C': [0.1, 1, 10, 100, 1000]},
                     scoring = 'neg_mean_squared_error',
                     cv = 10,
                     return_train_score = True)

model_c = svr_c.fit(X_train, Y_train.ravel())
print(model_c.best_params_)
grid_c_result = pd.DataFrame(data=model_c.cv_results_)

test_score = grid_c_result['mean_test_score']
test_std = grid_c_result['std_test_score']

train_score = grid_c_result['mean_train_score']
train_std = grid_c_result['std_train_score']

# Visualize the learning curve
x = [0.1, 1, 10, 100, 1000]
fig, ax = plt.subplots(figsize = (5, 5))

ax.plot(x, abs(train_score), '-o',  color = 'b',  markerfacecolor = "None", label = 'Train score')
ax.plot(x, abs(test_score), '-o', color = 'g', markerfacecolor = "None", label = 'Validation score')
ax.set_xscale('log')
ax.set_xlabel('Number of estimators')
ax.set_ylabel('Scores')
ax.legend()

#%% Fit the model with the optimal parameters
svr_opt = SVR(kernel = 'rbf', C = 1000, gamma = 0.01)

# Cross validation
svr_train_error, opt_cv_error = cross_validation(X_train, Y_train.ravel(), model = svr_opt, cv_method = kf)

# Make prediction for the test set and access the error
svr_opt.fit(X_train, Y_train.ravel())
pred = svr_opt.predict(X_test)
svr_test_error = np.sqrt(mean_squared_error(Y_test, pred))

# Print out model performance metrics
print(' Training error of {0:5.2f}'.format(svr_train_error))
print(' Validation error of {0:5.2f}'.format(opt_cv_error))
print(' Test error of {0:5.2f}'.format(svr_test_error))

#%% Parity plot to visualize prediction results
svr_train_pred = svr_opt.predict(X_train)
plt.scatter(y_train, svr_train_pred)
plt.scatter(y_test, pred)

plt.xlabel('Binding Energy from DFT (eV)')
plt.ylabel('Binding Energy from SVR (eV)')
plt.title('ML vs DFT Energy for SVR Model')
