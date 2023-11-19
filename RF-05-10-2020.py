#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import seaborn as sns


# In[5]:


from sklearn.datasets import load_digits


# In[6]:


digits=load_digits()


# In[7]:


digits


# In[8]:


digits.keys()


# In[9]:


print(digits.DESCR)


# In[10]:


digits.keys()


# In[11]:


digits.data


# In[12]:


digits.data.shape


# In[14]:


digits.data[0]


# In[18]:


digits.data[1]


# In[15]:


digits.target


# In[17]:


digits.target[0:40]


# In[21]:


digits.feature_names


# In[22]:


digits.target_names


# In[23]:


digits.images


# In[24]:


digits.images.shape


# In[25]:


import pylab as pl
pl.gray()
pl.matshow(digits.images[0])


# In[27]:


for i in range(5):
    pl.matshow(digits.images[i])


# In[28]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)


# In[31]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[35]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


clf= DecisionTreeClassifier()


# In[37]:


clf.fit(X_train,y_train);


# In[38]:


y_predict=clf.predict(X_test)


# In[39]:


y_predict


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix


# In[42]:


print(classification_report(y_test,y_predict))


# In[44]:


print(confusion_matrix(y_test,y_predict))


# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


rfc=RandomForestClassifier(n_estimators=60)


# In[49]:


rfc.fit(X_train,y_train);


# In[50]:


predictions=rfc.predict(X_test)


# In[51]:


print(classification_report(y_test,predictions))


# In[52]:


print(confusion_matrix(y_test,predictions))


# In[ ]:




