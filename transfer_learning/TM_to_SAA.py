# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:03:54 2022

@author: zongx
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
from sklearn.metrics import mean_squared_error
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

#%%
# Load TM dataset for training and SAA dataset for testing
TM_data = pd.read_excel('TM_Data.xlsx', sheet_name = 'TM_Reduced')
SAA_data = pd.read_excel('TM_Data.xlsx', sheet_name = 'SAA')

labels = list(TM_data.columns)
del labels[0:5]
del labels[-1]
descriptors = labels

# Select training data
X = TM_data.iloc[:, 5:15]
y_train = TM_data.iloc[:, 15]

# Standardize the training data to a unit variance and zero mean
scaler = StandardScaler().fit(X)
X_train = scaler.transform(X)

# Check if there have a unit variance and zero mean
Xs_std = np.std(X_train, axis = 0)
print('The std of each column in standardized X:')
print(Xs_std)
Xs_mean = np.mean(X_train, axis = 0)
print('The mean of each column standardized X:')
print(Xs_mean)

# Standardize the test data to a unit variance and zero mean
X_SAA = SAA_data.iloc[:, 5:15]
y_test = SAA_data.iloc[:, 15]

saa_scaler = StandardScaler().fit(X_SAA)
X_test = saa_scaler.transform(X_SAA)

# Check if there have a unit variance and zero mean
Xs_std_saa = np.std(X_test, axis = 0)
print('The std of each column in standardized X:')
print(Xs_std_saa)
Xs_mean_saa = np.mean(X_test, axis = 0)
print('The mean of each column standardized X:')
print(Xs_mean_saa)

#%% Set up cross-validation scheme                             
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

#%% Train XGBoost default model with training data
xgb_def = xgb.XGBRegressor(random_state=0)

def_train_error, def_cv_error = cross_validation(X_train, y_train,
                                                 model = xgb_def,
                                                 cv_method = kf)

xgb_def.fit(X_train, y_train)
def_pred = xgb_def.predict(X_test)
def_test_error = np.sqrt(mean_squared_error(y_test, def_pred))

# Print out default model performance metrics (Proved to match previous results)
print(' Training error of {0:5.2f}'.format(def_train_error))
print(' Validation error of {0:5.2f}'.format(def_cv_error))
print(' Test error of {0:5.2f}'.format(def_test_error))

#%%
xgb_opt = xgb.XGBRegressor(colsample_bytree = 0.7, # def:1; opt:0.7
                           subsample = 0.8, # 0.7
                           max_depth = 4, # opt:4
                           learning_rate = 0.3, # B4: 0.3
                           n_estimators = 100, # 100 // If this number is too large, the model is prone to overfitting!!! So just use default value = 100
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

#%% Feature importance analysis
explainer = shap.TreeExplainer(xgb_opt)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names = descriptors, plot_type = 'bar')
shap.summary_plot(shap_values, X_test, feature_names = descriptors)

#%% Parity plot to visualize ML prediction results
plt.scatter(y_train, opt_train_pred, color = '#51315e', label = 'Training')
plt.scatter(y_test, opt_pred, color = '#cf91a3', label = 'Test')

plt.xlabel('Binding Energy from DFT (eV)')
plt.ylabel('Binding Energy from XGBoost (eV)')
plt.title('ML vs DFT Energy for XGBoost Model', fontsize = 14)

plt.legend()
plt.savefig(fname = 'Visualization Results', dpi = 1200, bbox_inches = 'tight')

#%% Optimize XGBoost model with GridSearchCV
xgb_def = xgb.XGBRegressor(random_state=0)
param_grid = {
    #'colsample_bytree': np.linspace(0.1, 1.0, 10), # (0, 1] default: 1; optimum=0.3
    #'n_estimators': range(10, 500, 50), #opt=60
    #'max_depth': range(1, 11, 1), #opt=4
    #'gamma': np.linspace(0, 1, 10) # default = optimal = 0
    #'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 1.0] #opt=0.3
    #'subsample': np.linspace(0.1, 1.0, 10) # (0, 1] default: 1; opt=0.3
    #'reg_alpha': np.linspace(0.0, 1.0, 11) # default: 0; opt=0.1
    #'reg_lambda': np.linspace(0.0, 1.0, 11) # default: 1; opt=0.8
    }

clf = GridSearchCV(estimator = xgb_def,  # It doesn't matter which estimator model to use
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