# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 21:09:26 2021

@author: zongx
"""

# Import necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate, LeaveOneOut, train_test_split

#%%
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

#%% Data preprocessing
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

#%% Split the data into training and test set, set-up cross-validation
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
kfold = KFold(n_splits = 10, shuffle = True, random_state = 0)
loo = LeaveOneOut()

#%% Create helper functions for linear regression
def linear_regression(X, y):
    """Create linear regression object
    :param X: feature matrix
    :param y: response 
    :type y: 1D np array
    :return: regression model
    :rtype: sklearn object
    """   
    model = linear_model.LinearRegression(fit_intercept = True)
    model.fit(X, y)
    
    return model
    
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

def ols(x_train, y_train, x_test, y_test):
    model = linear_regression(x_train, y_train)
    n_features = x_train.shape[1]
    # Cross validation
    train_RMSE, cv_RMSE = cross_validation(x_train, y_train, model = model, cv_method = kfold)
    
    # Make prediction for the test set and access the error
    y_pred = model.predict(x_test)
    test_RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Print out performance metrics
    print('Model 0 has {0:} feature(s): '.format(n_features))      
    print(' Training error of {0:5.2f}'.format(train_RMSE))
    print(' Validation error of {0:5.2f}'.format(cv_RMSE))
    print(' Test error of {0:5.2f}'.format(test_RMSE))
    
    # Extract parameters from the model
    intercept = model.intercept_
    coef = model.coef_
    
    # Parity plot: actual data vs prediction
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred)
    plt.plot([-8, 0], [-8, 0], 'r', linewidth = 2)
    
    return test_RMSE, train_RMSE, cv_RMSE, intercept, coef
    
#%% Linear regression with each feature
test_RMSE = []
train_RMSE = []
cv_RMSE = []
intercepts = []
coefficients = []

for i in range(len(descriptors)):
    x_train = X_train[:, i].reshape(-1, 1)
    x_test = X_test[:, i].reshape(-1, 1)
    te, tr, cv, inter, co = ols(x_train, y_train, x_test, y_test)
    test_RMSE.append(te)
    train_RMSE.append(tr)
    cv_RMSE.append(cv)
    intercepts.append(inter)
    coefficients.append(co)

#%% Linear regression with all ten features
test, train, crossva, intercept, coef = ols(X_train, y_train, X_test, y_test)

#%%
# Parity plot to visualize prediction results
y_train_pred = intercept + np.dot(X_train, coef.reshape(-1, 1))
y_test_pred = intercept + np.dot(X_test, coef.reshape(-1, 1))

plt.scatter(y_train, y_train_pred)
plt.scatter(y_test, y_test_pred)

plt.xlabel('Binding Energy from DFT (eV)')
plt.ylabel('Binding Energy from OLS (eV)')
plt.title('ML vs DFT Energy for Multilinear Model')

# The prediction can also be calculated from the parameters
# Genral formula: y = intercept + np.dot(X, coef)
# coef = coef_0.reshape(-1, 1)
# y_predict = intercept_0 + np.dot(X_scaled, coef)

#%% Plot results re each descriptor as comparison
fig, ax = plt.subplots()
plt.xticks(rotation = 30)
ax.bar(descriptors, test_RMSE)
ax.set_ylim([0, 2])
ax.set_ylabel('Test RMSE')








