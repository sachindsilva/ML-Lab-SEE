# Write a program to implement the k-Nearest Neighbor classification algorithm on the Breast Cancer dataset and visualize the results.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report

data=pd.read_csv('Breast_Cancer.csv')

print("Dataset : ",data)

x=data.drop('diagnosis',axis=1)
x=x.iloc[:,:-1].values
y=data['diagnosis']

print("x : ",x)
print("y : ",y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

y_predict=knn.predict(x_test)

print("Predicted values : ",y_predict)


# Constructing confusion matrix and classification report


# CONFUSION MATRIX

cm=confusion_matrix(y_test,y_predict)

print("Confusion Matrix : ",cm)
print("\n")


cr=classification_report(y_test,y_predict)


print("Classification Report : ",cr)

for i in range(3):
    r1=np.where(y_predict == i)
    r2=np.where(y_test == i)

    if(i==0):
        m='*'
        c='ted'
    elif(i==1):
        m='o'
        c='green'
    elif(i==2):
        m='x'
        c='yellow'
    
    plt.plot(x_test[r1,1],x_test[r1,0],marker=m,color=c)










