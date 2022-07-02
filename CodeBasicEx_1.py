#!/usr/bin/env python
# coding: utf-8

# # Predict Home Price Using Linear Regression Algorithm

# In[35]:


import pandas as pd #used for data cleaning and analysis
import numpy as np # working with arrays. 
from sklearn import linear_model #LinearRegression model
import matplotlib.pyplot as plt #supports various types of graphical representations like Bar Graphs, Histograms, Line Graph, Scatter Plot, Stem Plots, etc


# In[5]:


#Read file from your systems
df = pd.read_csv('E:\\DataScience\\CSV_DB\\homeprices.csv')
df


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[12]:


new_df = df.drop('price',axis='columns')
new_df


# In[14]:


price = df.price
price


# In[16]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)


# In[18]:


reg.predict([[3300]])


# In[21]:


reg.coef_


# In[23]:


reg.intercept_


# In[25]:


reg.predict([[5000]])


# In[28]:


area_df = pd.read_csv("E:\\DataScience\\CSV_DB\\areas.csv")
area_df.head(3)


# In[30]:


p = reg.predict(area_df)
p


# In[32]:


area_df['prices']=p
area_df


# In[33]:


area_df.to_csv("prediction.csv")

