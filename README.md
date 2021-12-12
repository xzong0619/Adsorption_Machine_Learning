# Learning structure-sensitive scaling relations for small species adsorption on platinum surfaces
This repository contains the workflow of predicting adsorption energies based on physical features using various machine learning models. 

The developed ML model can rapidly screen potential catalyst materials, unify adsorbate binding into a single model, and account for the variation of the site type that escapes the traditional scaling relations.

## Developer
- Xue Zong (xzong@udel.edu)

## Dataset
The dataset includes binding energies of small adsorbates on various platinum surfaces calculated from density function theory (DFT) in Dataset.csv
- 16 types of adsorbates
- 12 types of Pt surfaces
- 295 data points

## Machine Learning Models Explored:
- Ordinary Least Square (OLS) regression
- Kernel Ridge Regression (KRR)
- Support Vector Regression (SVR)
- Extra Tree Regression (ETR)
- Random Forest Regression (RFR)
- Extreme Gradient Boosting regression (XGBoost)

## Dependencies
- [Numpy](https://numpy.org/): Used for vector and matrix operations
- [Matplotlib](https://matplotlib.org/): Used for plotting
- [Scipy](https://www.scipy.org/): Used for linear algebra calculations
- [Pandas](https://pandas.pydata.org/): Used to import data from Excel files
- [Sklearn](https://scikit-learn.org/stable/): Used for training machine learning models
- [Seaborn](https://seaborn.pydata.org/): Used for plotting

## Related Publication
__Zong, X.__; Vlachos, D. G. Unified Scaling Relations for Predicting Small Adsorbates Adsorption on Platinum Surfaces (2021). (Submitted)
