# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:17:36 2021

@author: zongx
"""
# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from statistics import mean
from sklearn.model_selection import KFold, cross_validate, train_test_split, cross_val_score

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

#%% Fit the model with the optimal parameters
krr_opt = KernelRidge(kernel = 'rbf', alpha = 0.001, gamma = 0.1)
model = krr_opt.fit(X_train, y_train)

# Set up cross-validation
scores = cross_val_score(model, X_train, y_train, scoring='r2', cv = 10)
cv_r2 = mean(scores)

# Make predictions
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

#%%
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)
