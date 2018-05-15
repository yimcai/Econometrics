#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 20:59:47 2018

@author: yimingcai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices

#%%
df = pd.read_excel("Test_Exercise1.xls", index_col = "Observation")
fig, ax = plt.subplots(1, 1)
xs = df.Advertising.tolist()
ys = df.Sales.tolist()
ax.scatter(xs, ys, s =10)
xs_bar = np.mean(xs)
ys_bar = np.mean(ys)

cov_xy = 0
cov_xx = 0
for i in range(len(xs)):
    cov_xy += xs[i]* (ys[i] -ys_bar)
    cov_xx += xs[i]*(xs[i] -xs_bar)
    
b = cov_xy/cov_xx
a = ys_bar - b*xs_bar
x0, x1 = 0, 20
y0 = a+ b*x0
y1 = a+ b*x1
ax.plot([x0, x1], [y0, y1], c = "red")
ax.set_xlim([0, 20])
ax.set_ylim([10, 60])
ax.set_xlabel("Advertising")
ax.set_ylabel("Sales")

Y,X = dmatrices("Sales ~ Advertising" ,data =df)
mod = sm.OLS(Y, X).fit()
print (mod.summary())
#%% Q2 standard error 
xs_ss = []
for i in range(len(xs)):
    xs_ss.append((xs[i] - xs_bar)**2)
ys_predicted = [a+ b*xs[i] for i in range(len(xs))]
error_ss = [(ys_predicted[i] - ys[i])**2 for i in range(len(xs))]
SE_b = np.sqrt(np.sum(error_ss)/(np.sum(xs_ss)*(len(xs)-2)))
#%% Q3 residuals 
residuals_df = pd.Series([ys[i] - a-b*xs[i] for i in range(len(xs))]) 

#%%Q5
df_drop = df.drop(12)
Y,X = dmatrices("Sales ~ Advertising" ,data =df_drop)
mod2 = sm.OLS(Y, X).fit()
print (mod2.summary())






