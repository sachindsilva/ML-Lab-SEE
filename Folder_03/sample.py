# Demonstrate the use of the Support Vector Machine algorithm for a 
# regression problem on the 'Iris flower dataset' and evaluate the 
# performance of the model.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data=pd.read_csv('Iris.csv')
# print("Dataset : ",data)
print("\n")

x=data['Id'].astype(int).values
print(x)

y=data['PetalLengthCm'].astype(int).values

print(y)

y=y.reshape(len(y),1)

print("Reshaped y : ",y)






