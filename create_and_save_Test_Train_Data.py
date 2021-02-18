'''create and save trainings data
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib


# === load data ===
my_str_path = str(pathlib.Path(__file__).parent.absolute())+'/kc_house_prices/King_County_House_prices_dataset.csv'
data = pd.read_csv(my_str_path) 

#data = pd.read_csv('kc_house_prices/King_County_House_prices_dataset.csv') 
data['sqft_basement_num'] = pd.to_numeric(data[data['sqft_basement']!='?'].sqft_basement)
data.head()

data = data.replace([np.inf, -np.inf], np.nan)
data.dropna(inplace=True)

# === built training data ===
Y = data['price']

X = data[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15', 'sqft_basement_num']]

X = pd.get_dummies(data=X, drop_first=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)

# %% Test save and load data
# define path
m_str_save_dict = str(pathlib.Path(__file__).parent.absolute())+'/kc_house_prices/'
m_str_ap = '.csv'

# save 
X_train.to_csv(m_str_save_dict+'X_train' +m_str_ap ,index=False) 
X_test.to_csv( m_str_save_dict+'X_test' +m_str_ap ,index=False)     
Y_train.to_csv(m_str_save_dict+'Y_train' +m_str_ap ,index=False)      
Y_test.to_csv( m_str_save_dict+'Y_test' +m_str_ap ,index=False)