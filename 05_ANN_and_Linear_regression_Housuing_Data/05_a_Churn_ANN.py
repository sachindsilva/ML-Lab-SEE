from tensorflow.python import train
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

data=pd.read_csv("https://raw.githubusercontent.com/amppmann/ML-Lab-SEE/master/Folder_05/Churn_Modelling.csv")
print("Dataset : ",data.head())


x=data.iloc[:,3:-1].values
y=data.iloc[:,-1].values

print("x : ",x)
print("x : ",y)


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

x[:,2]=le.fit_transform(x[:,2])
print("Label Encoded x : ",x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')

x=np.array(ct.fit_transform(x))
print("One Hot Encoded x : ",x)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Feature scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# Building the ann model

ann=tf.keras.models.Sequential()

# Adding the input layer and first hidden layer

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


# Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#Output Layer

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# Compile

ann.compile(optimizer='adam' ,loss='binary_crossentropy',metrics=['accuracy'])

ann.fit(x_train,y_train,batch_size=32,epochs=10)

y_predict=ann.predict(x_test)
y_predict=y_predict>0.5

print(np.concatenate((y_predict.reshape(len(y_predict),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_predict)
print('Confusion Matrix : ',cm)
print('Accuracy score of the model ',accuracy_score(y_test,y_predict))





