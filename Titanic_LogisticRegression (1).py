#!/usr/bin/env python
# coding: utf-8

# ## We are considering Titanic Dataset (The famous Titanic boat)
# Follow the below steps for doing Logistic Regression
# 1. Collect Data
# 2. Analyze Data
# 3. Data Wrangling (Data Cleaning involves removing Null values and unnecessary columns)
# 4. Train and Test Data
# 5. Accuracy Check and Logistic Regression

# ## Collect Data

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic_data = pd.read_csv('Titanic_Survival.csv')
titanic_data.head(10)


# In[5]:


print('No. of passangers in original dataset:', (len(titanic_data.index)))


# ## Analyze Data

# In[6]:


sns.countplot(x = 'Survived', data = titanic_data)


# In[7]:


sns.countplot(x ='Survived', hue ='Sex' ,data = titanic_data)


# In[8]:


sns.countplot(x ='Survived', hue ='Pclass', data = titanic_data)


# In[9]:


titanic_data['Age'].plot.hist()


# In[11]:


titanic_data['Fare'].plot.hist(bins = 20, figsize =(10,5))


# In[12]:


titanic_data.info()


# In[13]:


sns.countplot(x ='SibSp', data = titanic_data)


# In[15]:


sns.countplot(x = 'Parch', data = titanic_data)


# ## Data Wrangling

# In[54]:


'''Finding out the Null values in the dataset. True = null and False = not null'''
titanic_data.isnull()


# In[18]:


titanic_data.isnull().sum()


# From above, we can conclude that Age, Cabin and Embarked have Nulls.

# In[19]:


sns.boxplot(x = 'Pclass', y ='Age', data = titanic_data)


# In[23]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar = False)


# In[24]:


titanic_data.dropna(inplace = True)


# In[25]:


sns.heatmap(titanic_data.isnull(), yticklabels = False, cbar = False)


# In[26]:


titanic_data.isnull().sum()


# In[27]:


titanic_data.head(10)


# In[32]:


sex = pd.get_dummies(titanic_data['Sex'],drop_first=True)
sex.head(5)


# In[35]:


embarked = pd.get_dummies(titanic_data['Embarked'], drop_first= True)
embarked.head(5)


# In[36]:


pcl = pd.get_dummies(titanic_data['Pclass'], drop_first= True)
pcl.head(5)


# ## Concatenate all the new cleaned columns into the dataset

# In[37]:


titanic_data = pd.concat([titanic_data,sex,embarked,pcl],axis = 1)


# In[38]:


titanic_data.head(5)


# In[39]:


titanic_data.drop(['Sex','Name','Ticket','Embarked','Pclass'],axis = 1,inplace= True)


# In[40]:


titanic_data.head(10)


# In[41]:


titanic_data.drop(['Cabin'],axis = 1, inplace = True)


# In[42]:


titanic_data.head(5)


# ## Train Data

# In[44]:


X = titanic_data.drop('Survived',axis = 1)
y = titanic_data['Survived']


# In[52]:


import sklearn


# In[53]:


from sklearn.model_selection import train_test_split

