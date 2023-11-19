#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("spam - 1.csv")


# In[3]:


df


# In[4]:


df.groupby('Category').describe()


# In[5]:


df['spam']=df['Category'].apply(lambda x:1 if x=='spam'else 0)


# In[6]:


df.head()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam)


# In[9]:


X_train


# In[10]:


X_train.shape


# In[11]:


y_train


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer


# In[13]:


v=CountVectorizer()


# In[14]:


X_train_count=v.fit_transform(X_train)


# In[15]:


X_train_count.toarray()


# In[16]:


X_train_count.toarray().shape


# In[17]:


X_train_count.toarray()[:2]


# In[18]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()


# In[20]:


model.fit(X_train_count,y_train);


# In[21]:


emails=[
    'Hey, can we get together to watch football game tomorrow',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward'
]


# In[24]:


emails_count=v.transform(emails)


# In[25]:


model.predict(emails_count)


# In[26]:


X_test_count=v.transform(X_test)


# In[27]:


model.score(X_test_count,y_test)


# In[ ]:




