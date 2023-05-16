import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('Salary_Data.csv')
print("Dataset : ",data)


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


print("x : ",x)
print("y : ",y)



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


model=LinearRegression()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print("Predicted value : ",y_predict)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_predict,color='blue')
plt.title('Year of Experience Vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

