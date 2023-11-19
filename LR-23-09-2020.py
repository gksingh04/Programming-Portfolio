#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('homeprices.csv')


# In[5]:


df


# In[6]:


df.describe()


# In[8]:


plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red')


# In[9]:


X=df.drop('price',axis='columns')
X


# In[10]:


Y=df.price
Y


# In[11]:


from sklearn import linear_model


# In[12]:


reg=linear_model.LinearRegression()


# In[13]:


reg.fit(X,Y);


# In[14]:


reg.coef_        #m


# In[15]:


reg.intercept_       #c


# In[ ]:


# Price for 3300


# In[17]:


reg.predict([[3300]])


# In[18]:


#y=mx+c


# In[19]:


(reg.coef_)*3300+reg.intercept_ 


# In[20]:


reg.predict([[1000]])


# In[21]:


prediction=reg.predict(X)


# In[22]:


prediction


# In[23]:


plt.plot(X,Y,'r')
plt.plot(X,prediction,'b')


# In[24]:


from sklearn import metrics


# In[25]:


print(metrics.mean_absolute_error(Y, prediction))


# In[26]:


print(metrics.mean_squared_error(Y, prediction))


# In[27]:


print(np.sqrt(metrics.mean_squared_error(Y, prediction)))


# In[ ]:




