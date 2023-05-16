import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')

print("Dataset : ",data.head())

x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values


print("x : ",x)
print("y : ",y)

y=y.reshape(len(y),1)

print("Reshaped y : ",y)


# Feature scaling

from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()
sc_y=StandardScaler()

x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)



# Training the SVR model using training set

from sklearn.svm import SVR

regressor=SVR(kernel='rbf')

regressor.fit(x,y)

print("New Result")

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1) ))

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y).reshape(-1,1),color='red')

plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)),color='blue')

plt.title('Truth or Bluff (SVR)')

plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
