# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 18:10:26 2021

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
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'out'

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

#%% Build the model using default hyperparameters
# Instantiate an XGBoost regressor object using default values
xgb_reg = xgb.XGBRegressor(random_state = 0)

# Cross validation (Return train_error and cross-validate error)
reg_train_error, reg_cv_error = cross_validation(X_train, y_train,
                                                 model = xgb_reg,
                                                 cv_method = kf)

# Make prediction for the test set and access the error
xgb_reg.fit(X_train, y_train)
reg_pred = xgb_reg.predict(X_test)
reg_test_error = np.sqrt(mean_squared_error(y_test, reg_pred))

# Print out default model performance metrics
print(' Training error of {0:5.2f}'.format(reg_train_error))
print(' Validation error of {0:5.2f}'.format(reg_cv_error))
print(' Test error of {0:5.2f}'.format(reg_test_error))

# Check default model parameters
# xgb_reg

#%% Fit the model with the optimal parameters
xgb_opt = xgb.XGBRegressor(colsample_bytree = 1, # def:1
                           subsample = 0.7, # 0.7
                           max_depth = 6, # 3
                           learning_rate = 0.3, # B4: 0.3
                           n_estimators = 100, 
                           reg_alpha = 0, # B4: 0
                           reg_lambda = 1, # B4: 1
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

# Check default model parameters
# xgb_opt

#%% Parity plot to visualize ML prediction results
plt.scatter(y_train, opt_train_pred, color = '#51315e', label = 'Training')
plt.scatter(y_test, opt_pred, color = '#cf91a3', label = 'Test')

plt.xlabel('Binding Energy from DFT (eV)')
plt.ylabel('Binding Energy from XGBoost (eV)')
plt.title('ML vs DFT Energy for XGBoost Model', fontsize = 14)

plt.legend()
plt.savefig(fname = 'Visualization Results', dpi = 1200, bbox_inches = 'tight')

#%% Feature Importances
# XGBoost Built-in Feature Importance. Same API interface like other sklearn models
sorted_idx = xgb_opt.feature_importances_.argsort()

plt.barh(np.array(descriptors)[sorted_idx], xgb_opt.feature_importances_[sorted_idx], color = '#9a5b88')
plt.xlabel("XGBoost Feature Importance (Gain)")

plt.savefig(fname = 'Feature Importance', dpi = 1200, bbox_inches = 'tight')


# Check the type of importance
# xgb_opt.importance_type

#%%
# XGBoost feature importances based on Weight
plot_importance(xgb_opt)

#%%
# Feature importance computed with SHAP values
explainer = shap.TreeExplainer(xgb_opt)
shap_values = explainer.shap_values(X_test)

y_names = ['CN', 'n_metal', 'GCN', 'Valency', 'Electroneg', 'Bader Charge',
           'd-band center', 'd-band width', 'n_H',  'n_H_NN']

shap.summary_plot(shap_values, X_test, feature_names = y_names, plot_type = 'bar')
shap.summary_plot(shap_values, X_test, feature_names = y_names)

#%%
# Remove the most important feature (valency) and rerun the optimal model to see its performance
X_train_no_va = np.delete(X_train, 3, axis = 1)
X_test_no_va = np.delete(X_test, 3, axis = 1)

# Take the optimal model and return performance metrics
no_va_train, no_va_cv = cross_validation(X_train_no_va, y_train, model = xgb_opt, cv_method = kf)

# Make prediction for the test set and access the error
xgb_opt.fit(X_train_no_va, y_train)
no_va_pred = xgb_opt.predict(X_test_no_va)
no_va_test = np.sqrt(mean_squared_error(y_test, no_va_pred))

# Print out default model performance metrics
print(' Training error of {0:5.2f}'.format(no_va_train))
print(' Validation error of {0:5.2f}'.format(no_va_cv))
print(' Test error of {0:5.2f}'.format(no_va_test))

# Plot updated feature importances after removing valency feature
no_va_idx = xgb_opt.feature_importances_.argsort()
no_va_des = ['CN', 'n_metal', 'GCN', 'chi', 'Bader', 'ed', 'wd', 'n_H', 'n_H_NN']
 
plt.barh(np.array(no_va_des)[no_va_idx], xgb_opt.feature_importances_[no_va_idx])
plt.xlabel("Xgboost Feature Importance w/o Valency")              

#%%
# Remove four least important features (Wd, n_H_NN, GCN, ed) and rerun the model
descriptors = ['CN', 'n_metal', 'GCN', 'Valency', 'chi', 'Bader',
               'ed', 'wd', 'n_H', 'n_H_NN']

remove_idx = [2, 6, 7, 9]

X_train_re = np.delete(X_train, remove_idx, axis = 1)
X_test_re = np.delete(X_test, remove_idx, axis = 1)

# Take the optimal model and return performance metrics
re_train, re_cv = cross_validation(X_train_re, y_train, model = xgb_opt, cv_method = kf)

# Make prediction for the test set and access the error
xgb_opt.fit(X_train_re, y_train)
re_pred = xgb_opt.predict(X_test_re)
re_test = np.sqrt(mean_squared_error(y_test, re_pred))

# Print out default model performance metrics
print(' Training error of {0:5.2f}'.format(re_train))
print(' Validation error of {0:5.2f}'.format(re_cv))
print(' Test error of {0:5.2f}'.format(re_test))

# Plot updated feature importances after removing valency feature
re_des = np.delete(descriptors, remove_idx)
re_idx = xgb_opt.feature_importances_.argsort()
 
plt.barh(np.array(re_des)[re_idx], xgb_opt.feature_importances_[re_idx])
plt.xlabel("Xgboost Feature Importance Removing Unimportant Features")     

#%% Code not used
"""
Try early stopping (NOT USEFUL! n_estimators too small)
xgb_model = xgb.XGBRegressor(colsample_bytree = 0.8, 
                             subsample = 0.7,
                             max_depth = 4,
                             learning_rate = 0.1)

xgb_model.fit(X_train, y_train, early_stopping_rounds = 5,
              eval_set = [(X_test, y_test)])

"""

#%% Hyperparameter searching by each

# Model default: objective (reg:squarederror); eval_metric (rmse for regression)
param_grid = {
    #'colsample_bytree': np.linspace(0.1, 1.0, 10), # (0, 1] default: 1
    #'n_estimators': range(10, 100, 10),
    #'max_depth': range(1, 11, 1),
    #'gamma': np.linspace(0, 0.5, 6)
    #'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 1.0]
    #'subsample': np.linspace(0.1, 1.0, 10) # (0, 1] default: 1
    #'reg_alpha': np.linspace(0.0, 1.0, 11) # default: 0
    #'reg_lambda': np.linspace(0.0, 1.0, 10) # default: 1
    }

clf = GridSearchCV(estimator = xgb_reg,  # It doesn't matter which estimator model to use
                   param_grid = param_grid,
                   scoring = 'neg_mean_squared_error',
                   cv = 10, return_train_score = True)

model = clf.fit(X_train, y_train)
grid_result = pd.DataFrame(data=model.cv_results_)
print("Best: %f using %s" % (model.best_score_, model.best_params_))

test_score = grid_result['mean_test_score']
test_std = grid_result['std_test_score']

train_score = grid_result['mean_train_score']
train_std = grid_result['std_train_score']

#%%
# Visualize the learning curve
x = np.linspace(0.0, 1.0, 10)
fig, ax = plt.subplots(figsize = (5, 5))

ax.plot(x, abs(train_score), '-o',  color = 'b',  markerfacecolor = "None", label = 'Train score')
ax.plot(x, abs(test_score), '-o', color = 'g', markerfacecolor = "None", label = 'Validation score')
ax.set_xlabel('Lambda')
ax.set_ylabel('Scores: MSE')
ax.legend()
"""
#%% Gird Search (Run for too long. Not FINISHED!)
"""
xgb_model = xgb.XGBRegressor(random_state = 0)

param_grid = {
    'colsample_bytree': np.linspace(0.1, 1.0, 10),
    'learning_rate': np.linspace(0.1, 1.0, 10),
    'subsample': np.linspace(0.1, 1.0, 10),
    'max_depth': range(1, 11, 1)
    }

clf = GridSearchCV(estimator = xgb_model, 
                   param_grid = param_grid,
                   scoring = 'neg_mean_squared_error',
                   cv = 10, return_train_score = True)

model = clf.fit(X_train, y_train)
grid_result = pd.DataFrame(data=model.cv_results_)
print("Best: %f using %s" % (model.best_score_, model.best_params_))

test_score = grid_result['mean_test_score']
test_std = grid_result['std_test_score']

train_score = grid_result['mean_train_score']
train_std = grid_result['std_train_score']
"""

#%%
"""
#2 Permutation based feature importance with sklearn
# Randomly shuflle each feature and compute the change in model's performances
# The features which impact the performance the most are the most importance ones
perm_importance = permutation_importance(xgb_opt, X_test, y_test)

perm_sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(np.array(descriptors)[perm_sorted_idx], perm_importance.importances_mean[perm_sorted_idx])
plt.xlabel("Permutation Importance")
"""