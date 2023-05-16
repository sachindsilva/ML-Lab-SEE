import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report



data=pd.read_csv('diabetes.csv')
print("Dataset : ",data.head())


x=data.drop('Outcome',axis=1)
y=data['Outcome']

print("Feature values : ",x.head())
print("Target values : ",y.head())


# Splitting the dataset into training and testing set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



rfc=RandomForestClassifier(n_estimators=5,max_features=5)
rfc.fit(x_train,y_train)

y_rfc=rfc.predict(x_test)

print("Predicted value : ",y_rfc)


# Evaluation of the model using confusion matrix and classification report

cm=confusion_matrix(y_test,y_rfc)

print("--Confusion Matrxix--")
print(cm)
print("\n")

target_names=["Diabetes","Normal"]
cr=classification_report(y_test,y_rfc,target_names=target_names)

print("--Classification Report--")
print(cr)
print("\n")

index=np.arange(0,len(y_test))

fig,ax=plt.subplots(1,1,figsize=(15,5))

plt.scatter(index,y_test,c='red',label='Actual value')
plt.scatter(index,y_rfc,c='blue',label='Predicted value')

plt.legend()
plt.show()
