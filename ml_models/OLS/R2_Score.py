# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:54:55 2021

@author: zongx
"""
# Import necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statistics import mean
from sklearn.model_selection import KFold, cross_validate, LeaveOneOut, train_test_split, cross_val_score

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

# Set up cross-validation scheme                                  
kfold = KFold(n_splits = 10, shuffle = True, random_state = 0)

#%%
model = linear_model.LinearRegression(fit_intercept = True)
model.fit(X_train, y_train)

# Set up cross-validation
scores = cross_val_score(model, X_train, y_train, scoring='r2', cv = 10)
cv_r2 = mean(scores)

# Make predictions
y_pred = model.predict(X_test)

intercept = model.intercept_
coef = model.coef_

y_train_pred = intercept + np.dot(X_train, coef.reshape(-1, 1))

#%%
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)
