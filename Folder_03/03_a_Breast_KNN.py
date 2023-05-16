import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets



dataset=datasets.load_breast_cancer()

data=dataset.data 
target=dataset.target

print("Data : ",data)
print("Target : ",target)

x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2)


knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

y_predict=knn.predict(x_test)

print("Predicted value : ",y_predict)



for i in range(3):
    r1=np.where(y_predict == i)
    r2=np.where(y_test == i)

    if i==0:
        m='*'
        c='red'
    elif i==1:
        m='o'
        c='green'
    elif i==2:
        m='x'
        c='blue'
    plt.scatter(x_test[r1,1],x_test[r1,0],marker=m,color=c)      
plt.show()
