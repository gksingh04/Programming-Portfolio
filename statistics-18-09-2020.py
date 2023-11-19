#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import statistics


# In[6]:


m=np.random.randint(7,10,21)
m


# In[7]:


Mean=statistics.mean(m)


# In[8]:


Mean


# In[9]:


Median=statistics.median(m)


# In[10]:


Median


# In[11]:


mode=statistics.mode(m)
mode


# In[14]:


n=np.random.randn(30)


# In[15]:


n


# In[16]:


#Range=Max(n)-Min(n)


# In[17]:


Range=np.max(n)-np.min(n)


# In[18]:


Range


# In[19]:


Q1=np.percentile(n,25)
Q1


# In[21]:


Q2=np.percentile(n,50)
Q2


# In[22]:


Q3=np.percentile(n,75)
Q3


# In[24]:


IQR=Q3-Q1
IQR


# In[25]:


v=np.var(n)


# In[26]:


v


# In[27]:


s=np.std(n)
s


# In[1]:


#normal distribution


# In[2]:


from numpy import random


# In[4]:


x=random.normal()
x


# In[5]:


x=random.normal(size=(2,3))


# In[6]:


x


# In[7]:


x=random.normal(size=(2,3))


# In[8]:


x


# In[25]:


x=random.normal(loc=1,scale=5,size=(1000))
x


# In[26]:


import matplotlib.pyplot as plt


# In[27]:


import seaborn as sns


# In[28]:


sns.distplot(x,hist=False)


# In[29]:


population=random.normal(size=100000)


# In[30]:


population


# In[31]:


sns.distplot(population)


# In[33]:


import numpy as np
np.mean(population)


# In[36]:


sample1=np.random.choice(population,10000)
sample2=np.random.choice(population,10000)
sample3=np.random.choice(population,10000)
sample4=np.random.choice(population,10000)
sample5=np.random.choice(population,10000)


# In[37]:


all_samples=[sample1,sample2,sample3,sample4,sample5]
sample_mean=[]
for i in all_samples:
    sample_mean.append(np.mean(i))


# In[38]:


sample_mean


# In[39]:


np.mean(sample_mean)


# In[1]:


#Entropy


# In[2]:


from scipy.stats import entropy


# In[4]:


p=[9/14,5/14]
E=entropy(p,base=2)
print(E)


# In[5]:


p=[1/6,1/6,1/6,1/6,1/6,1/6]
E=entropy(p,base=2)
print(E)


# In[1]:


# confusion Matrix


# In[4]:


import pandas as pd


# In[5]:


data = {'Actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        }


# In[9]:


df=pd.DataFrame(data, columns=['Actual','Predicted'])
df


# In[6]:


from sklearn.metrics import confusion_matrix,classification_report


# In[15]:


print(confusion_matrix(df['Predicted'],df['Actual']))


# In[16]:


print(classification_report(df['Predicted'],df['Actual']))


# In[ ]:




