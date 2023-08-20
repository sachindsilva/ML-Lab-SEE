import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter


data=pd.read_csv('Google_stock.csv')

# print("Dataset : ",data.head())

# x=data.iloc[:,2].values

x=data.iloc[:,0].str.replace('/','').str.replace('-','').astype('int').values
y=data.iloc[:,-1].str.replace(',','').astype('int').values

x=x.reshape(len(x),1)
y=y.reshape(len(y),1)
print("x : ",x)
print("y : ",y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# from sklearn.preprocessing import StandardScaler

# sc=StandardScaler()

# x_train=sc.fit_transform(x_train).astype('int')
# x_test=sc.transform(x_test).astype('int')


model=LinearRegression()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print("Predicted y : ",y_predict)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_predict,color='blue')
plt.title('High Vs Volume')
date_form = DateFormatter("%m-%d")
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

