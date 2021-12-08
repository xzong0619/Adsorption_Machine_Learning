# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:17:24 2021

@author: xzong
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import graphviz
import shap
import matplotlib.pyplot as plt
import matplotlib
from xgboost import plot_importance
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['figure.dpi'] = 600.

#%%
# Load the dataset
data = pd.read_excel('Dataset.xlsx')


descriptors = ['CN', 'n_metal', 'GCN', 'Valency', 'chi', 'Bader',
               'ed', 'wd', 'n_H', 'n_H_NN']

# Select data for input and output
X = data.iloc[:, [2, 4, 5, 6, 7, 8, 9, 10, 13, 14]]

"""
features = ['Valency', 'Bader', 'n_metal', 'chi', 'GCN', 'ed',  
            'CN', 'n_H', 'wd', 'n_H_NN']

"""
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

#%% Fit the model with the optimal parameters
xgb_opt = xgb.XGBRegressor(colsample_bytree = 1, # Opt: 0.8
                           subsample = 0.7,
                           max_depth = 6,
                           learning_rate = 0.3, # Opt: 0.1
                           n_estimators = 100, 
                           reg_alpha = 0, # Opt: 0.4
                           reg_lambda = 1, # Opt: 0.78
                           random_state = 0)

# Cross validation (Return train_error and cross-validate error)
opt_train_error, opt_cv_error = cross_validation(X_train, y_train,
                                                 model = xgb_opt,
                                                 cv_method = kf)

# Make prediction for the test set and access the error
xgb_opt.fit(X_train, y_train)
opt_train_pred = xgb_opt.predict(X_train)
opt_pred = xgb_opt.predict(X_test)
opt_test_error = np.sqrt(mean_squared_error(y_test, opt_pred))

# Print out default model performance metrics
print(' Training error of {0:5.2f}'.format(opt_train_error))
print(' Validation error of {0:5.2f}'.format(opt_cv_error))
print(' Test error of {0:5.2f}'.format(opt_test_error))

#%%
#descriptors = ['CN', 'n_metal', 'GCN', 'Valency', 'chi', 'Bader',
#               'ed', 'wd', 'n_H', 'n_H_NN']

remove_idx = [[9], [7, 9], [7, 8, 9], [0, 7, 8, 9], [0, 6, 7, 8, 9],
              [0, 2, 6, 7, 8, 9], [0, 2, 4, 6, 7, 8, 9], 
              [0, 1, 2, 4, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6, 7, 8, 9]]

remove_train = []
remove_cv = []
remove_test = []

for i in range(len(remove_idx)):
    re_idx = remove_idx[i]
    
    x_train_re = np.delete(X_train, re_idx, axis = 1)
    x_test_re = np.delete(X_test, re_idx, axis = 1)
    
    re_train, re_cv = cross_validation(x_train_re, y_train, model = xgb_opt,
                                       cv_method = kf)
    
    xgb_opt.fit(x_train_re, y_train)
    re_pred = xgb_opt.predict(x_test_re)
    re_test = np.sqrt(mean_squared_error(y_test, re_pred))
    
    remove_train.append(re_train)
    remove_cv.append(re_cv)
    remove_test.append(re_test)

#%%
num_features = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

remove_train = [opt_train_error] + remove_train
remove_cv = [opt_cv_error] + remove_cv
remove_test = [opt_test_error] + remove_test

#%%
x = np.arange(10)

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])

ax.bar(x + 0.00, remove_train, color = '#51315e', width = 0.25, tick_label = num_features, label = 'Training')
ax.bar(x + 0.25, remove_cv, color = '#9a5b88', width = 0.25, tick_label = num_features, label = 'CV Validation')
ax.bar(x + 0.50, remove_test, color = '#cf91a3', width = 0.25, tick_label = num_features, label = 'Test')

plt.ylabel('RMSE (eV)')
plt.xlabel('Number of Features included in XGBoost')
plt.legend()