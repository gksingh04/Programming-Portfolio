#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


from matplotlib import pyplot as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data=pd.read_csv('Decision_Tree_ Dataset -1.csv')


# In[6]:


data.head()


# In[7]:


data.info


# In[8]:


X=data.values[:,0:5]
X


# In[9]:


Y=data.values[:,5]


# In[10]:


Y


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=42)


# In[13]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


clf=DecisionTreeClassifier()


# In[16]:


clf.fit(X_train,y_train);


# In[17]:


y_predict=clf.predict(X_test)


# In[18]:


y_predict


# In[19]:


from sklearn.metrics import classification_report,confusion_matrix


# In[20]:


print(classification_report(y_test,y_predict))


# In[21]:


print(confusion_matrix(y_test,y_predict))


# In[22]:


import seaborn as sns


# In[23]:


cm=confusion_matrix(y_test,y_predict)


# In[26]:


plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='.0f')
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[ ]:




