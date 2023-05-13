import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



data=pd.read_csv('diabetes.csv')

print("Dataset : ",data)

print("Shape of dataset : ",data.shape)


x=data.drop('Outcome',axis=1)
y=data["Outcome"]

print("Features set : ",x)
print("Target set : ",y)


# Splitting the dataset into training and testing sets


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Using the RFC Model

rfc=RandomForestClassifier(n_estimators=5,max_features=5)

rfc.fit(x_train,y_train)

y_predict=rfc.predict(x_test)


print("Predicted value : ",y_predict)


# Evaluation of model

from sklearn.metrics import classification_report,confusion_matrix

print("---Confusion Matrix---")

cm=confusion_matrix(y_test,y_predict)

print(cm)

print('\n')

print("---Classification Report---")


target_names=["Diabetes","Normal"]
cr=classification_report(y_test,y_predict,target_names=target_names)

print(cr)
print('\n')

index=np.arange(0,len(y_test))

fig,ax=plt.subplots(1,1,figsize=(15,5))

plt.scatter(index,y_test,c='red',label='True Value')

plt.scatter(index,y_predict,c='blue',label='Predicted Value')

plt.legend()
plt.show()




