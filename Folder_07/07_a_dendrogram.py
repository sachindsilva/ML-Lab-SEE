# Apply Hierarchical clustering on the Mall_Customers dataset and 
# visualize the clusters and plot the dendrograms.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('Mall_Customers.csv')

print("Dataset : ",data)




x=data.iloc[:,[3,4]].values

# Using dendrogram to find the optimal number of clusters..

import scipy.cluster.hierarchy as sch

dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')

plt.show()


# Training the Hierarchical Clustering model on the dataset

from sklearn.cluster import AgglomerativeClustering


hc=AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='ward')


y_hc=hc.fit_predict(x)

# Visualizing the clusters..


plt.scatter(x[y_hc == 0,0],x[y_hc == 0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[y_hc == 1,0],x[y_hc == 1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[y_hc == 2,0],x[y_hc == 2,1],s=100,c='green',label='Cluster 3')
plt.scatter(x[y_hc == 3,0],x[y_hc == 3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(x[y_hc == 4,0],x[y_hc == 4,1],s=100,c='magenta',label='Cluster 5')


plt.title("Clusters of Customers")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()













