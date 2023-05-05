#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('mcdonalds.csv')


# DATA EXPLORATION

# In[3]:


print("Column names in McDonalds dataset:-\n")
print(data.columns)


# In[4]:


print("Dimension of the dataset:-\n")
print(data.shape)


# In[5]:


print('Head of the dataset:-\n')
print(data.head())


# In[6]:


print("***selecting first 11 columns for our analysis***\n")
x=data.iloc[:,:11]


# In[7]:


print("Yes will be converted to True & No into False\n")
x = (x=='Yes')
print(x.head())


# In[8]:


print("Mean of the 11 selected columns:-\n")
print(round(x.mean(),2))


# In[9]:


# R> MD.pca <- prcomp(MD.x)
# R> summary(MD.pca)

# R> print(MD.pca, digits = 1)
#    Standard deviations (1, .., p=11):
#    Rotation (n x k) = (11 x 11):


# In[10]:


import numpy as np
from sklearn.decomposition import PCA

pcaX = PCA().fit(x)


# In[11]:


pcaX.explained_variance_ratio_


# In[12]:


pcaX.singular_values_


# In[13]:


pcaX.components_


# In[14]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[15]:


pct=PCA().fit_transform(x)


# In[16]:


kmeans = KMeans(n_clusters=5).fit(pct)
plt.scatter(pct[:, 0], pct[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title(" Principal components analysis of the fast food data set")
plt.show()


# SEGMENT EXTRACTION

# In[20]:


#CODE IN R
#Segment extraction using mixture of distribution
# R> library("flexmix")
# R> set.seed(1234)
# R> MD.m28 <- stepFlexmix(MD.x ~ 1, k = 2:8, nrep = 10,
# + model = FLXMCmvbinary(), verbose = FALSE)
# R> MD.m28

# Call:
# stepFlexmix(MD.x ~ 1, model = FLXMCmvbinary(),
# k = 2:8, nrep = 10, verbose = FALSE)

# R> plot(MD.m28,
# + ylab = "value of information criteria (AIC, BIC, ICL)")


# SEGMENT PROFILING

# In[21]:


from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage


# In[27]:


MD_k4 = KMeans(n_clusters=4, n_init=10, verbose=False).fit(x)


# In[38]:


from scipy.spatial.distance import squareform, pdist


# In[39]:


MD_kmeans = KMeans(n_clusters=8, n_init=10, verbose=0).fit_transform(x)
MD_kmeans = MD_kmeans.astype(int)
MD_k4 = MD_kmeans[3]
MD_vclust = linkage(squareform(pdist(x.T)))
plt.bar(range(len(MD_k4)), height=1, width=0.5, bottom=MD_k4, color='gray', alpha=0.5)
plt.xticks(range(len(MD_k4)), MD_vclust['ivl'][::-1], rotation=90)
plt.show()


# SEGMENT DESCRIBING

# In[ ]:


# Code in R
# mosiac plot for segment describing

# R> k4 <- clusters(MD.k4)
# R> mosaicplot(table(k4, mcdonalds$Like), shade = TRUE,
# + main = "", xlab = "segment number")


# SEGMENT EVALUATION PLOT

# In[42]:



# R> plot(visit, like, cex = 10 * female,
# + xlim = c(2, 4.5), ylim = c(-3, 3))
# R> text(visit, like, 1:4)

plt.scatter(visit, like, s=10*np.array(female), alpha=0.5)
plt.xlim(2, 4.5)
plt.ylim(-3, 3)

for i, txt in enumerate(range(1, 5)):
    plt.annotate(txt, (visit[i], like[i]))

plt.show()


# In[ ]:




