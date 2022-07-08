# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:37:23 2022

@author: xzong
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import graphviz
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from xgboost import plot_importance
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

# Set plotting parameters
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

#%% Preprocessing data
# Load dataset
new_data = pd.read_excel('Dataset.xlsx', sheet_name = 'New')

descriptors = ['CN', 'n_metal', 'GCN', 'Valency', 'chi_ME',
               'Bond_count', 'chi_NN', 'mass']

# Select input data
X = new_data.iloc[:, 3:11]

# Select binding energy as response
Y = new_data.iloc[:, 11]

# Standardize the data to a unit variance and zero mean
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

#Set up cross-validation scheme                                  
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

#%% Default XGB model; default model behaves better than the optimal model
xgb_reg = xgb.XGBRegressor(random_state = 0)

full_train_error, full_cv_error = cross_validation(X_train, y_train,
                                         model = xgb_reg,
                                         cv_method = kf)

xgb_reg.fit(X_train, y_train)
full_pred = xgb_reg.predict(X_test)
full_test_error = np.sqrt(mean_squared_error(y_test, full_pred))

# Print out default model performance metrics (Proved to match previous results)
print(' Training error of {0:5.2f}'.format(full_train_error))
print(' Validation error of {0:5.2f}'.format(full_cv_error))
print(' Test error of {0:5.2f}'.format(full_test_error))


#%% Fit the model with the optimal parameters
xgb_opt = xgb.XGBRegressor(colsample_bytree = 0.7, # def:1; opt = 0.7
                           subsample = 1, # 0.7
                           max_depth = 5, # 6
                           learning_rate = 0.3, # B4: 0.3
                           n_estimators = 70, # 100
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
opt_test_mae = mean_absolute_error(y_test, opt_pred)
opt_test_r2 = r2_score(y_test, opt_pred)

# Print out default model performance metrics
print(' Training error of {0:5.2f}'.format(opt_train_error))
print(' Validation error of {0:5.2f}'.format(opt_cv_error))
print(' Test error of {0:5.2f}'.format(opt_test_error))
print(' Test MAE of {0:5.2f}'.format(opt_test_mae))
print(' Test R2 of {0:5.2f}'.format(opt_test_r2))

#%% Feature importance analysis with SHAP values
explainer = shap.TreeExplainer(xgb_opt)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names = descriptors, plot_type = 'bar')
shap.summary_plot(shap_values, X_test, feature_names = descriptors)

#%% Feature importance built in XGBoost evaluated on Gain
sorted_idx = xgb_opt.feature_importances_.argsort()

plt.barh(np.array(descriptors)[sorted_idx], xgb_opt.feature_importances_[sorted_idx], color = '#9a5b88')
plt.xlabel("XGBoost Feature Importance (Gain)")

plt.savefig(fname = 'Feature Importance', dpi = 1200, bbox_inches = 'tight')

#%% XGBoost feature importance based on weight
plot_importance(xgb_opt)

#%%
# Manually plot the feature importance
descriptors = ['GCN', 'CN','mass', 'Valency', 'n_metal', 'Bond_count',
               'chi_ME', 'chi_NN']

y = [288, 229, 202, 179, 165, 143, 106, 69]

fig, ax = plt.subplots()
fig, ax = plt.subplots()

y_pos = np.arange(len(descriptors))

ax.barh(y_pos, y, align = 'center', color = '#9a5b88')
ax.set_yticks(y_pos)
ax.set_yticklabels(descriptors)

ax.invert_yaxis()
ax.set_xticks([])
ax.set_xlabel('XGBoost Feature Importance (Weight)')

plt.show()

#%% Parity plot to visualize prediction results
opt_train_pred = xgb_opt.predict(X_train)
plt.scatter(y_train, opt_train_pred, color = '#51315e', label = 'Training')
plt.scatter(y_test, opt_pred, color = '#cf91a3', label = 'Test')

plt.xlim([-8, 0])
plt.ylim([-8, 0])
plt.legend()

plt.xlabel('Binding Energy from DFT (eV)')
plt.ylabel('Binding Energy from XGBoost (eV)')
plt.title('ML vs DFT Energy for XGBoost Model')


#%% Hyperparameter searching by each; Results are consistent since parameters are searched by CV error
# Model default: objective (reg:squarederror); eval_metric (rmse for regression)
param_grid = {
    #'colsample_bytree': np.linspace(0.1, 1.0, 10), # (0, 1] default: 1; optimum=0.7
    #'n_estimators': range(10, 100, 10), #opt=60
    #'max_depth': range(1, 11, 1), #opt=4
    #'gamma': np.linspace(0, 1, 10) # default = optimal = 0
    #'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 1.0] #opt=0.3
    #'subsample': np.linspace(0.1, 1.0, 10) # (0, 1] default: 1; opt=0.7
    #'reg_alpha': np.linspace(0.0, 1.0, 11) # default: 0; opt=0.2
    #'reg_lambda': np.linspace(0.0, 1.0, 11) # default: 1
    }

clf = GridSearchCV(estimator = xgb_reg,  # It doesn't matter which estimator model to use
                   param_grid = param_grid,
                   scoring = 'neg_root_mean_squared_error',
                   cv = 10, return_train_score = True)

model = clf.fit(X_train, y_train)
grid_result = pd.DataFrame(data=model.cv_results_)
print("Best: %f using %s" % (model.best_score_, model.best_params_))

test_score = grid_result['mean_test_score']
test_std = grid_result['std_test_score']

train_score = grid_result['mean_train_score']
train_std = grid_result['std_train_score']
