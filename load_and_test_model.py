import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score


import pickle

# load data
m_str_save_dict = str(pathlib.Path(__file__).parent.absolute())+'/kc_house_prices/'
m_str_ap = '.csv'

X_train = pd.read_csv(m_str_save_dict+'X_train' +m_str_ap) 
X_test  = pd.read_csv( m_str_save_dict+'X_test' +m_str_ap)     
Y_train = pd.read_csv(m_str_save_dict+'Y_train' +m_str_ap)      
Y_test  = pd.read_csv( m_str_save_dict+'Y_test' +m_str_ap)

# load model
filename = 'my_linear_regression_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)



# Make predictions using the testing set
Y_test_pred = loaded_model.predict(X_test)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, Y_test_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_test, Y_test_pred))

# %% plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(Y_test_pred, Y_test, alpha=0.5, color='orchid')
fig.suptitle('Y_test_pred = f(X_test) vs Y_test ')
ax.axis('equal')
ax.set_ylabel("Y_test");
ax.set_xlabel("Y_test_pred = f(X_test)");
plt.show()