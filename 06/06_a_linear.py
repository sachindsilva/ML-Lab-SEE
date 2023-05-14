# Demonstrate the application of Simple Linear regression to predict 
# the stock market prices of any organization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv('Google_stock.csv')

print("Dataset : ",data)

x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

print("x : ",x)
print("y : ",y)




