# Demonstrate the use of the Support Vector Machine algorithm for a 
# regression problem on the 'Iris flower dataset' and evaluate the 
# performance of the model.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data=pd.read_csv('Iris.csv')
print("Dataset : ",data)
print("\n")

x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values


print("X : ",x)
print("\n")
print("Y : ",y)
print("\n")


y=y.reshape(len(y),1)

# print("Reshaped y : ",y)

# Feature Scaling

from sklearn.preprocessing import StandardScaler,LabelEncoder

sc_x=StandardScaler()
sc_y=StandardScaler()



x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)



print("Transformed x : ",x)
print("Transformed y : ",y)


# Training the SVR Model on the whole dataset

from sklearn.svm import SVR

regressor=SVR(kernel='rbf')

regressor.fit(x,sc_y)

# Predicting the result

print("New value")

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(1,-1)))


# # Visualizing the results

# plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y).reshape(-1,1),color='red')
# plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)),color='blue')

# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()





