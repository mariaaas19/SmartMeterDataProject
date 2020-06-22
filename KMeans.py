#!/usr/bin/env python
# coding: utf-8

# In[35]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from scipy.spatial.distance import cdist

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_excel(r'C:\Users\Maria Munir Stokes\Desktop\Smart Meter Data\KMeansData.xlsx',7)
data.shape
#data.head() #checking the data imported correctly
features = data.columns.tolist()#assigning the column names to a list
#print('data')
#print(data)
#print(type(features))
#print('features:',features)

#scaling data
scaled_data = StandardScaler().fit_transform(data)
#print('scaled_data')
#print(scaled_data)
scaled_data.shape
#display(scaled_data[0]) #each element of the array contains 6 normalised values for the household - 1 for each feature 
print('hey')

#elbow plot
wcss = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(scaled_data)
    kmeanModel.fit(scaled_data)
    
    centroids =kmeanModel.cluster_centers_#an array with the coordinates of the cluster's centers
    wcss.append(sum(np.min(cdist(scaled_data, centroids, 'euclidean'), axis=1)) / scaled_data.shape[0])
 #plot customisation
plt.plot(K, wcss)
plt.xlabel('K')
plt.ylabel('WCSS')
plt.title('Elbow Plot')
plt.show()

#k means clustering
kmeans = KMeans(n_clusters=5)
#kmeansmodel = kmeans.fit(scaled_data)
kmeansmodel = kmeans.fit(scaled_data)
#print(kmeansmodel)
centers = kmeansmodel.cluster_centers_
data['ClusterAssignment'] = kmeansmodel
# display(data) #another check
#data.to_excel(r"C:\Users\Maria Munir Stokes\Desktop\Smart Meter Data\control Clusters win resxxx1.xlsx",sheet_name = 'Cluster')
# #n of clusters, n of features
# #centroids_array = kmeansmodel.cluster_centers_
# #print('centroids')
# #display(centroids_array)
# #print('shape',centroids_array.shape) #5 clusters, 6 features

# features.append('feature_centroids')
# #iteration through the array of centroids
# #every cluster has a center in each dimension
# centroid_values = [np.append(a,index) for index, a in enumerate(kmeansmodel.cluster_centers_)]
# #creating dataframe for values that are plotted

# graph_df = pd.DataFrame(centroid_values,columns = features)
# #int value allows it to be plotted
# int_graph_df = graph_df['feature_centroids'].astype(int) 
# graph_df['feature_centroids'] = int_graph_df


#print('graph df:')
display(graph_df)

print('z:',z)
#figure plot
plt.figure(figsize = (15,5))
parallel_coordinates(graph_df,'feature_centroids',color = ['b', 'r', 'c', 'y','k'],marker = 'x')
plt.title('Results of K Means Control',fontweight = 'bold',fontsize = '14')
plt.xlabel('Features',fontweight = 'bold',fontsize = '11')
plt.xticks(fontweight = 'bold')
plt.ylabel('Centroid Values',fontweight = 'bold',fontsize = '11')

print('finish')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




