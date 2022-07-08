# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:48:10 2022

@author: xzong
"""

# Import necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, LeaveOneOut, train_test_split

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

#%% Data Preprocessing
# Load the dataset
data = pd.read_excel('Dataset.xlsx')
# Return column names

descriptors = ['CN', 'n_metal', 'GCN', 'Valency', 'chi_ME',
               'Bond_count', 'chi_NN', 'mass']

# Select input data
X = data.iloc[:, 3:11]

# Select binding energy as response
Y = data.iloc[:, 11]

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
    test_MAE = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Print out performance metrics
    print('Model 0 has {0:} feature(s): '.format(n_features))      
    print(' Training error of {0:5.2f}'.format(train_RMSE))
    print(' Validation error of {0:5.2f}'.format(cv_RMSE))
    print(' Test error of {0:5.2f}'.format(test_RMSE))
    print(' Test MAE of {0:5.2f}'.format(test_MAE))
    print(' Test R2 of {0:5.2f}'.format(test_r2))
    
    # Extract parameters from the model
    intercept = model.intercept_
    coef = model.coef_
    
    # Parity plot: actual data vs prediction
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred)
    plt.plot([-8, 0], [-8, 0], 'r', linewidth = 2)
    
    return test_RMSE, train_RMSE, cv_RMSE, test_MAE, test_r2, intercept, coef

#%% Linear regression with all ten features
test, train, test_mae, test_r2_score, crossva, intercept, coef = ols(X_train, y_train, X_test, y_test)

#%%
# Parity plot to visualize prediction results
y_train_pred = intercept + np.dot(X_train, coef.reshape(-1, 1))
y_test_pred = intercept + np.dot(X_test, coef.reshape(-1, 1))

plt.scatter(y_train, y_train_pred)
plt.scatter(y_test, y_test_pred)

plt.xlim([-8, 0])
plt.ylim([-8, 0])

plt.xlabel('Binding Energy from DFT (eV)')
plt.ylabel('Binding Energy from OLS (eV)')
plt.title('ML vs DFT Energy for Multilinear Model')

#%% Plot feature importance from their coefficients
coef_list = coef.tolist()

for i in range(len(coef_list)):
    val = coef_list[i]
    if val < 0:
        coef_list[i] = abs(val)

features = np.array(descriptors)

new_coef = np.array(coef_list)

sorted_idx = new_coef.argsort()
plt.barh(features[sorted_idx], new_coef[sorted_idx])
plt.xlabel("Absolute Coefficients")

plt.savefig("MLR Feature Importance.png", dpi=1200)
