#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create and save linear regression model
"""

import pandas as pd
from sklearn import linear_model
import pickle
import pathlib

# load data
m_str_save_dict = str(pathlib.Path(__file__).parent.absolute())+'/kc_house_prices/'
m_str_ap = '.csv'

X_train = pd.read_csv(m_str_save_dict+'X_train' +m_str_ap) 
X_test  = pd.read_csv( m_str_save_dict+'X_test' +m_str_ap)     
Y_train = pd.read_csv(m_str_save_dict+'Y_train' +m_str_ap)      
Y_test  = pd.read_csv( m_str_save_dict+'Y_test' +m_str_ap)

# create model
my_model = linear_model.LinearRegression() # Do not use fit_intercept = False if you have removed 1 column after dummy encoding
my_model.fit(X_train, Y_train)
# save the model to disk
filename = 'my_linear_regression_model.sav'
pickle.dump(my_model, open(filename, 'wb'))