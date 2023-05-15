import numpy as np
import pandas as pd

#Import corresponding library

# --> Splitting the dataset
from sklearn.model_selection import train_test_split

# --> To convert non-numerical to numerical values
from sklearn.preprocessing import LabelEncoder

# --> To Apply Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier 

# --> For constructing tree
from sklearn import tree

# --> To compute the accuracy of the model.
from sklearn import metrics


# Importing and using the dataset 'tennis.csv'

data=pd.read_csv('tennis.csv')

print("Dataset : ",data)

all_cols=data.columns

features=all_cols[1:5]

print("Features : ",features)

# Encoding all values of the dataset --> for non-numerical values..

for label in data.columns:
    data[label]=LabelEncoder().fit_transform(data[label])


# Input Values

input=data.iloc[:,:-1].values

target=data.iloc[:,-1].values

print("Input : ",input)
print("Target : ",target)


# Splitting the dataset into training and testing sets

x_train,x_test,y_train,y_test=train_test_split(input,target,test_size=0.2)

id3=DecisionTreeClassifier()

id3=id3.fit(x_train,y_train)

# Predicting the values of the model

y_predict=id3.predict(x_test)

print("Predicted values : ",y_predict)

# Determining the accuracy produced by the model

print("Accuracy of the model :",metrics.accuracy_score(y_test,y_predict))


# Plotting the model through tree

tree.plot_tree(id3,feature_names=features)


