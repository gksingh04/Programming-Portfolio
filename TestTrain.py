#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


print(sklearn.__version__)


# In[3]:


#!pip install sklearn


# In[4]:


import numpy as np


# In[14]:


X,y= np.arange(20).reshape((10,2)),range(10)


# In[15]:


X


# In[16]:


y


# In[17]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


# In[36]:


X_train


# In[32]:


X_test


# In[22]:


y_train


# In[23]:


y_test


# In[ ]:




