

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Social_Network_Ads.csv')

print("Dataset : ",data)

x=data.iloc[:,[0,1]].values
y=data.iloc[:,2].values

print("x : ",x)
print("y : ",y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# Feature Scaling..

from sklearn.preprocessing import StandardScaler


sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Fitting the Naive Bayes model to the training set


from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()

classifier.fit(x_train,y_train)

y_predict=classifier.predict(x_test)

print("Predicted y : ",y_predict)


# Constructing the confusion matrix and determining the accuracy of the model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm=confusion_matrix(y_test,y_predict)

print("--CONFUSION MATRIX")
print(cm)

print("ACCURACY OF THE MODEL = ",accuracy_score(y_test,y_predict))


# Visualising the results..

from matplotlib.colors import ListedColormap


# FOR TRAINING SETS

x_set,y_set=x_train,y_train

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(x1.min(),x2.max())
plt.ylim(x1.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],color=ListedColormap(('red','green'))(i),label=j)

plt.title('Naive Bayes (Training Set)')
plt.xlabel('Age')
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


# FOR TESTING SETS

x_set,y_set=x_test,y_test

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(x1.min(),x2.max())
plt.ylim(x1.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],color=ListedColormap(('red','green'))(i),label=j)

plt.title('Naive Bayes (Testing Set)')
plt.xlabel('Age')
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()







