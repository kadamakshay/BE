

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("sales_data_sample.csv",encoding='Latin-1')


# In[3]:


df.head()


# In[4]:


#information of the data
df.info()


# In[5]:


#get the statistical info of the data

df.describe()


# In[6]:


#check for null values
df.isnull().sum()


# In[7]:


#correlation matrix

plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)
plt.show()


# ### K means Clustering

# In[8]:


from sklearn.cluster import KMeans


# In[9]:


X=df[["PRICEEACH","SALES"]]


# In[10]:


wcss=[]
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(X)
    wcss.append(km.inertia_)
    
plt.figure(figsize=(12,6))
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss,color="green",linewidth=2)
plt.xlabel("K")
plt.ylabel("WCSS")
plt.grid()
plt.show()


# In[11]:


km_model=KMeans(n_clusters=4)
km_model.fit(X)
y_pred = km_model.predict(X)


# In[12]:


X['Target']=y_pred
X.head()


# In[13]:


sns.scatterplot(X.PRICEEACH,X.SALES, hue=X.Target,palette=['red','orange','blue','green'])
plt.title("Price of Each vs Total Sales")
plt.show()


# ### Hierarchical clustering

# In[14]:


#Find the optimal k value using dendrogram graph

import scipy.cluster.hierarchy as shc 
dendro = shc.dendrogram(shc.linkage(X, method="ward"))  
plt.title("Dendrogram Plot")  
plt.ylabel("Euclidean Distances")  
plt.xlabel("Sales")  
plt.show()  


# In[15]:


#train tge model

from sklearn.cluster import AgglomerativeClustering  
hc= AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
y_pred= hc.fit_predict(X.drop("Target",axis='columns'))  


# In[16]:


X = X.drop("Target",axis='columns')


# In[17]:


X = X.values


# In[18]:


X


# In[19]:


plt.figure(figsize=(10,7))
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c="red",label="Cluster 1")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c="blue",label="Cluster 2")
plt.title("Clusters of Sales")
plt.xlabel("PRICEEACH")
plt.xlabel("SALES")
plt.show()


# In[ ]:




