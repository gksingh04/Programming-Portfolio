#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


cancer


# In[5]:


print(cancer.DESCR)


# In[6]:


print(type(cancer)) #no


# In[7]:


cancer.keys()


# In[8]:


cancer.data.shape


# In[9]:


cancer.data[0]


# In[10]:


#cancer.taget[0:10]
cancer.target.shape


# In[11]:


cancer.target


# In[12]:


cancer.target[0:20]


# In[13]:


cancer.target_names


# In[14]:


cancer.feature_names


# In[15]:


# Create a dataframe with the four feature variables
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)


# In[16]:


df


# In[17]:


cancer['feature_names']


# In[18]:


cancer['feature_names'].shape


# In[19]:


# Create a dataframe with the four feature variables
X=df = pd.DataFrame(cancer.data, columns=cancer.feature_names)


# In[20]:


df.info()


# In[21]:


cancer['target'].shape


# In[22]:


Y=df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
#df_target
Y


# In[23]:


df.head()


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


#X_train, X_test, y_train, y_test = train_test_split(df, np.ravel(df_target), test_size=0.30, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=101)


# In[26]:


from sklearn.svm import SVC


# In[27]:


model = SVC()


# In[28]:


model.fit(X_train,y_train);


# In[29]:


predictions = model.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report,confusion_matrix


# In[31]:


print(confusion_matrix(y_test,predictions))


# In[32]:


print(classification_report(y_test,predictions))


# In[33]:


model = SVC(C=0.1, gamma=1, kernel='rbf')
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))


# In[34]:


model = SVC(C=1, gamma=0.1, kernel='rbf')
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))


# In[35]:


model = SVC(C=100, gamma=0.001, kernel='rbf')
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))


# In[36]:


model = SVC(C=1000, gamma=0.0001, kernel='rbf')
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))


# In[37]:


model = SVC(C=1, gamma=0.0001, kernel='rbf')
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))


# In[ ]:




