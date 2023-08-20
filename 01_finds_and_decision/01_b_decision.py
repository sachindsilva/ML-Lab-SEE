import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data=pd.read_csv('Tennis.csv')

print("Dataset : ",data)


all_cols=data.columns
features=all_cols[1:5]

for i in data.columns:
    data[i]=LabelEncoder().fit_transform(data[i])

inputs=data.iloc[:,:-1].values

target=data.iloc[:,-1].values

print("Inputs : ",inputs)
print("Target : ",target)
print("\n")


x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2,random_state=0)

id3=DecisionTreeClassifier()

id3.fit(x_train,y_train)

y_predict=id3.predict(x_test)

print("Predicted value : ",y_predict)


print("Accuracy score of the model : ",accuracy_score(y_test,y_predict))

tree.plot_tree(id3,feature_names=features)
