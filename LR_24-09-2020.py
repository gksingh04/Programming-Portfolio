#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("insurance_data.csv")


# In[3]:


df


# In[4]:


plt.scatter(df.age,df.bought_insurance,color='red',marker='+')


# In[5]:


X=df.drop('bought_insurance',axis='columns')


# In[6]:


X


# In[7]:


Y=df.bought_insurance
Y


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)


# In[10]:


X_test


# In[11]:


X_train


# In[12]:


y_train


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


reg=LinearRegression()
reg.fit(X,Y);


# In[15]:


prediction=reg.predict(X)
prediction


# In[16]:


plt.scatter(df.age,df.bought_insurance, color='red',marker='+')
plt.plot(X,prediction,'b')


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


model=LogisticRegression()
model.fit(X_train,y_train);


# In[19]:


prediction=model.predict(X_test)


# In[20]:


prediction


# In[21]:


y_test


# In[22]:


y_predict=model.predict([[35]])
y_predict


# In[23]:


from sklearn.metrics  import classification_report, confusion_matrix


# In[24]:


print(confusion_matrix(y_test,prediction))


# In[25]:


print(classification_report(y_test,prediction))


# In[ ]:




