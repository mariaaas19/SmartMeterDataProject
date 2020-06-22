#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
sns.set
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_excel(r'C:\Users\Maria Munir Stokes\Desktop\Smart Meter Data\KMeansData.xlsx',2)
scaled_data = preprocessing.scale(data) #scaling and centering data

pca = PCA() #creates an object
pca.fit(scaled_data) #all the math for the pca is carried out
pca_data = pca.transform(scaled_data) # generate components for pca graph

pca_df = pd.DataFrame(pca_data[0:,0:],index=[i for i in range(pca_data.shape[0])],columns = ['PC'+str(i) for i in range(pca_data.shape[1])])
display(pca_df)
#scree plot
per_var = np.round(pca.explained_variance_ratio_*100,decimals =1 )  #percentage that each variation accounts for
labels = ['PC'+ str(x) for x in range(1,len(per_var)+1)] #scree plot labels
percentage_df = pd.DataFrame(columns = pca.explained_variance_ratio_*100)
print('percentage:')
display(per_var)


#barplot
plt.figure(figsize=(6,6))
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
#creating new data frame with 80% of data
pc123 = pca_df[['PC0','PC1','PC2']].copy()
display(pc123)

#elbow plot
wcss = []
for i in range (1,10):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(pc123)
    wcss.append(kmeans_pca.inertia_)
#
plt.figure(figsize=(6,6))
plt.plot(range(1,10),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method Plot')

#k means clustering
kmeans = KMeans(n_clusters=5)
kmeansmodel = kmeans.fit_predict(pc123)
print(kmeansmodel)

pc123['ClusterAssignment'] = kmeansmodel
display(pc123) #another check

cluster0 = pc123[pc123['ClusterAssignment'] == 0]
cluster1 = pc123[pc123['ClusterAssignment'] == 1]
cluster2 = pc123[pc123['ClusterAssignment'] == 2]
cluster3 = pc123[pc123['ClusterAssignment'] == 3]
cluster4 = pc123[pc123['ClusterAssignment'] == 4]

print(0)
display(cluster0)

threed = plt.figure(figsize = (10,6))
threed = plt.axes(projection ='3d')
zline = np.linspace(0,3)
xline = np.sin(zline)
xline = np.linspace(0,3)
yline = np.cos(zline)
yline = np.linspace(0,3)


threed.scatter3D(cluster0.PC0,cluster0.PC1,cluster0.PC2,color = 'green')
threed.scatter3D(cluster1.PC0,cluster1.PC1,cluster1.PC2,color = 'orange')
threed.scatter3D(cluster2.PC0,cluster2.PC1,cluster2.PC2,color = 'cyan')
threed.scatter3D(cluster3.PC0,cluster3.PC1,cluster3.PC2,color = 'black')
threed.scatter3D(cluster4.PC0,cluster4.PC1,cluster4.PC2,color = 'pink')
threed.set_xlabel('PC0')
threed.set_ylabel('PC1')
threed.set_zlabel('PC2')
threed.set_title('Clusters',fontweight='bold')

pc123.to_excel(r"C:\Users\Maria Munir Stokes\Desktop\Smart Meter Data\PCA ClustersHELLOO summmm res.xlsx",sheet_name = 'Cluster')

#######No Resident data
data_nores = pd.read_excel(r'C:\Users\Maria Munir Stokes\Desktop\Smart Meter Data\KMeansData.xlsx',7)
scaled_data_nores=preprocessing.scale(data_nores)
#repeating the scree plot without the resident values
pca.fit(scaled_data_nores)
pca_data_nores = pca.transform(scaled_data_nores)
per_var_nores = np.round(pca.explained_variance_ratio_*100,decimals =1 )
labels = ['PC'+ str(x) for x in range(1,len(per_var_nores)+1)]
print('pca no res',pca_data_nores)
##print('pca type', pca_data_nores.type) array
#creating dataframe for pca no res data
pca_nores_df = pd.DataFrame(pca_data_nores[0:,0:],index=[i for i in range(pca_data_nores.shape[0])],columns = ['PC'+str(i) for i in range(pca_data_nores.shape[1])])
display(pca_nores_df)
#bar plot
plt.figure(figsize=(6,6))
plt.bar(x=range(1,len(per_var_nores)+1),height=per_var_nores,tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot (No Resident Data)')
plt.show()

print('percentage (no residents):')
display(per_var_nores)

pc123_nores = pca_nores_df[['PC0','PC1','PC2']].copy()
display(pc123_nores)


#elbow plot
wcss = []
for i in range (1,10):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(pc123_nores)
    wcss.append(kmeans_pca.inertia_)
#
plt.figure(figsize=(6,6))
plt.plot(range(1,10),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method Plot (No Resident Data)')

#k means clustering
kmeans = KMeans(n_clusters=5)
kmeansmodel = kmeans.fit_predict(pc123_nores)
print(kmeansmodel)

pc123_nores['ClusterAssignment'] = kmeansmodel
display(pc123_nores) #another check

cluster0n = pc123_nores[pc123_nores['ClusterAssignment'] == 0]
cluster1n = pc123_nores[pc123_nores['ClusterAssignment'] == 1]
cluster2n = pc123_nores[pc123_nores['ClusterAssignment'] == 2]
cluster3n = pc123_nores[pc123_nores['ClusterAssignment'] == 3]
cluster4n = pc123_nores[pc123_nores['ClusterAssignment'] == 4]
print(0)
display(cluster0n)

threedn = plt.figure(figsize = (10,6))
threedn= plt.axes(projection ='3d')
zline = np.linspace(0, 0.01, 2)
xline = np.sin(zline)
xline = np.linspace(0,0.01,3)
yline = np.cos(zline)
yline = np.linspace(0,0.01,3)


threedn.scatter3D(cluster0n.PC0,cluster0n.PC1,cluster0n.PC2,color = 'green')
threedn.scatter3D(cluster1n.PC0,cluster1n.PC1,cluster1n.PC2,color = 'black')
threedn.scatter3D(cluster2n.PC0,cluster2n.PC1,cluster2n.PC2,color = 'cyan')
threedn.scatter3D(cluster3n.PC0,cluster3n.PC1,cluster3n.PC2,color = 'pink')
threedn.scatter3D(cluster4n.PC0,cluster4n.PC1,cluster4n.PC2,color = 'orange')
threedn.set_xlabel('PC0')
threedn.set_ylabel('PC1')
threedn.set_zlabel('PC2')
threedn.set_title('Clusters - No Resident Data',fontweight='bold')

pc123_nores.to_excel(r"C:\Users\Maria Munir Stokes\Desktop\Smart Meter Data\PCA Clusters summmm no res.xlsx",sheet_name = 'Cluster')


# In[ ]:





# In[ ]:




