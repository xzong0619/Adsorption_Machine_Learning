# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:37:47 2021

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

#%% Fit the model with the optimal parameters
xgb_opt = xgb.XGBRegressor(colsample_bytree = 1, # Opt: 0.8
                           subsample = 0.7,
                           max_depth = 6,
                           learning_rate = 0.3, # Opt: 0.1
                           n_estimators = 100, 
                           reg_alpha = 0, # Opt: 0.4
                           reg_lambda = 1, # Opt: 0.78
                           random_state = 0)

#descriptors = ['CN', 'n_metal', 'GCN', 'Valency', 'chi', 'Bader',
#               'ed', 'wd', 'n_H', 'n_H_NN']

re_idx = [0, 6, 7, 8, 9]

# Remaining descriptors: n_metal, GCN, valency, chi, Bader

x_train_re = np.delete(X_train, re_idx, axis = 1)
xgb_opt.fit(x_train_re, y_train)

#%% Retrieve the standard scaler
mean = scaler.mean_[1:6]
variance = scaler.var_[1:6]
std_dev = np.sqrt(variance)

#%% Compare DFT results and ML prediction
x_ch2 = [7.33, 6.67, 5.78, 5.39, 5.44, 5.39, 2.83, 3.11, 3.17, 4]
y_ch2 = [-4.170688, -4.662551, -4.667886, -4.578436, -4.4544, 
         -4.683179, -4.586207, -4.437673, -4.512873, -4.310429]

gcn_ch2 = []
for i in range(len(x_ch2)):
    gcn_energy = 0.01 * x_ch2[i] - 4.55
    gcn_ch2.append(gcn_energy)


ml_ch2_data = []

for i in range(len(x_ch2)):
    gcn = x_ch2[i]
    ml_ch2 = [2, gcn, 2, 2.55, 4.2835]
    ml_ch2_data.append(ml_ch2)

ml_ch2_stand = (ml_ch2_data - mean) / std_dev
ml_pred = xgb_opt.predict(ml_ch2_stand)

#%%
plt.scatter(y_ch2, gcn_ch2, label = 'GCN Results')
plt.scatter(y_ch2, ml_pred, label = 'ML Results')

plt.legend()
plt.plot([-4.7, -4.15], [-4.7, -4.15], c = 'k')
plt.xlabel('DFT Calculated BE (eV)')
plt.ylabel('Predicted BE (eV)')

#%%
plt.scatter(x_ch2, gcn_ch2, label = 'GCN Results')
plt.scatter(x_ch2, ml_pred, label = 'ML Results')
plt.scatter(x_ch2, y_ch2, label = 'DFT Results')

plt.legend(loc = 'upper right')
plt.xticks([2, 3, 4, 5, 6, 7, 8])
plt.xlabel('GCN')
plt.ylabel('Binding Energy (eV)')

#%% Generate data for prediction
GCN_data = np.linspace(2, 8, num = 20)

CH3_data = []
CH2_data = []

for i in range(20):
    add_gcn = GCN_data[i]
    add_CH3 = [1, add_gcn, 1, 2.55, 4.1985]
    CH3_data.append(add_CH3)
    
    add_CH2 = [2, add_gcn, 2, 2.55, 4.2835]
    CH2_data.append(add_CH2)

CH3_stand = (CH3_data - mean) / std_dev
CH2_stand = (CH2_data - mean) / std_dev

CH3_pred = xgb_opt.predict(CH3_stand)
CH2_pred = xgb_opt.predict(CH2_stand)

#%% Visualize the prediction results
x_ch3 = [7.5, 6.67, 5.83, 5.5, 5.5, 5.5, 2.5, 3, 2.92, 3.58, 3.5, 4.42]
y_ch3 = [-2.24297, -2.334887, -2.392736, -2.305122, -2.28812, -2.300841,
         -2.224483, -2.336209, -2.409126, -2.352042, -2.620518, -2.322212]

plt.scatter(GCN_data, CH3_pred)
plt.scatter(x_ch3, y_ch3)

#%%
diff = CH3_pred - CH2_pred
plt.scatter(GCN_data, diff)

plt.xticks([2, 3, 4, 5, 6, 7, 8])

plt.xlabel('GCN')
plt.ylabel('BE_CH3 - BE_CH2 (eV)')

#%%
x_ch2 = [7.33, 6.67, 5.78, 5.39, 5.44, 5.39, 2.83, 3.11, 3.17, 4]
y_ch2 = [-4.170688, -4.662551, -4.667886, -4.578436, -4.4544, 
         -4.683179, -4.586207, -4.437673, -4.512873, -4.310429]

plt.scatter(GCN_data, CH2_pred)
plt.scatter(x_ch2, y_ch2)

#%% Test case
# Use CH2* on Pt(111) as a test
x_test = [2, 7.33, 2, 2.55, 4.2835]
x_stand = (x_test - mean) / std_dev

# xgb_opt.predict(np.array(x_stand).reshape(1, -1)) = -4.1712503 # real DFT value is -4.170688


