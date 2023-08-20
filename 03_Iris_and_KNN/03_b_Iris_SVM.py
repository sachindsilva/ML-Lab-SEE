import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import datasets


data=pd.read_csv('Iris.csv')
print("Dataset : ",data)

x=data.iloc[:,[0]].values
y=data.iloc[:,1].values

print("x : ",x)

y=y.reshape(len(y),1)
print("y : ",y)

#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()
sc_y=StandardScaler()

x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)


# Training the SVM Model using training dataset

from sklearn.svm import SVR 

regressor=SVR(kernel='rbf')
regressor.fit(x,y)

print("New Value")
y_predict=sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

print("Predicted value : ",y_predict)


plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y).reshape(-1,1),color='red')

plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)),color='blue')

plt.title('Iris - ID VS SepalLength (SVR)')
plt.xlabel('ID')
plt.ylabel('Sepal Length (cm)')
plt.show()



