# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Read data
prop = "color"          # the property to predict

dat_train = pd.read_csv("Data/IllustrisTNG300-1-train.csv", header=0)
dat_test = pd.read_csv("Data/IllustrisTNG300-1-val.csv", header=0)

# list of features
#features = ["M200c", "sigma", "c200c", "z_form", "s", "q", "M_acc", "E_s", "Mean_vel", "f_mass_cen", "R0p9", "spin"]
features = ["M200c", "sigma", "c200c", "z_form", "s", "q", "M_acc", "E_s", "R0p9", "spin"]

# Get x and y
x_train = dat_train[features]
y_train = dat_train[[prop]]
print(x_train.head())

x_test = dat_test[features]
y_test = dat_test[[prop]]

# standardize the data
x_test = (x_test-x_train.mean())/x_train.std()
x_train = (x_train-x_train.mean())/x_train.std()

y_test = y_test.to_numpy().reshape(-1,)
y_train = y_train.to_numpy().reshape(-1,)

# add constant column
x_train = sm.add_constant(x_train.to_numpy())

model = sm.OLS(endog=y_train, exog=x_train).fit()
print(model.summary())

# write performance measures to file
with open("Results/featureselection.txt", "a") as f:
    f.write(prop)
    f.write("\t")
    f.write(" ".join(features))
    f.write("\n")
    f.write(model.summary().as_text())
    f.write("\n")
    f.write("\n")
