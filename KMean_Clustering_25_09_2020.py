#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('income.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


plt.xlabel('Age')
plt.ylabel('Income')
plt.scatter(df['Age'],df['Income'])


# In[6]:


X=df.drop('Name',axis='columns')


# In[7]:


X


# In[8]:


from sklearn.cluster import KMeans


# In[9]:


km=KMeans(n_clusters=3)


# In[12]:


km.fit(X);


# In[13]:


y_predict=km.predict(X)


# In[14]:


y_predict


# In[15]:


df['cluster']=y_predict
df


# In[16]:


km.cluster_centers_


# In[22]:


df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
plt.scatter(df0['Age'],df0['Income'],color='green')
plt.scatter(df1['Age'],df1['Income'],color='red')
plt.scatter(df2['Age'],df2['Income'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*', label='centrod')


# In[23]:


new=km.predict([[41,50000]])


# In[24]:


new


# In[25]:


from sklearn.preprocessing import MinMaxScaler


# In[27]:


scaler=MinMaxScaler()
df['Income']=scaler.fit_transform(df[['Income']])
df['Age']=scaler.fit_transform(df[['Age']])
df


# In[28]:


X=df.values[:,1:3]


# In[29]:


km=KMeans(n_clusters=3)
km.fit(X);


# In[30]:


y_predict=km.fit_predict(X)
y_predict


# In[32]:


df['cluster']=y_predict
df


# In[33]:


df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
plt.scatter(df0['Age'],df0['Income'],color='green')
plt.scatter(df1['Age'],df1['Income'],color='red')
plt.scatter(df2['Age'],df2['Income'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*', label='centrod')


# In[ ]:




