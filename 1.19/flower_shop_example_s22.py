#!/usr/bin/env python
# coding: utf-8

# # What does a Data Science project look like?

# #### Client: 
# 
# My Flower Shop
# 
# #### Ask: 
# 
# Need to send out a catalog. Would like to know who to send it to.
# 
# #### One translation: 
# 
# Can we classify people according to whether they're likely to reorder?

# ## Notebook Setup

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

sns.set_theme()

#%matplotlib inline


# In[3]:


#%run?


# ## Read in Data

# In[3]:


#datafile_name = '../data/flowershop_data.csv'
datafile_name = 'D:/【CU】2022 Spring/5293_py/5293py 1.19/flowershop_data.csv'


# In[5]:


# print out the first few lines of the datafile, looks like standard csv with a header and dates
#!head -5 {datafile_name}
datafile_name

#!more flowershop_data.csv P 5


# In[6]:


# read in data, note the number of rows and columns
df = pd.read_csv('D:/【CU】2022 Spring/5293_py/5293py 1.19/flowershop_data.csv', header=0, parse_dates=True)
print('{:d} rows, {:d} columns'.format(*df.shape))


# In[15]:


# use head to make sure the data was read correctly
df.head(10)


# ## Evaluate and Clean Variables

# In[6]:


# use info to evaluate missing values and datatypes
df.info()


# ### Evaluate and Clean: Stars

# In[7]:


# plot histogram for stars
sns.catplot(x='stars',kind='count',data=df);
plt.title('Distribution of Stars Variable');


# In[8]:


# an awful lot of 5s
# get highest name counts
df[df.stars == 5].lastname.value_counts().head()


# In[24]:


df[df.stars == 5].lastname.value_counts().head()


# In[9]:


# get differences in mean stars by name
robinson_mean = df[df.lastname == 'ROBINSON'].stars.mean()
other_mean = df[df.lastname != 'ROBINSON'].stars.mean()
robinson_mean_diff = robinson_mean - other_mean
print('robinson mean stars: {}'.format(robinson_mean))
print('other mean stars   : {}'.format(other_mean))
print('difference in mean : {}'.format(robinson_mean_diff))


# In[26]:


df[df.lastname == 'ROBINSON'].head()


# In[10]:


# get counts for names
n_robinson = len(df[df.lastname == 'ROBINSON'])
n_other = len(df) - n_robinson
print('n_robinson: {}'.format(n_robinson))
print('n_other   : {}'.format(n_other))


# In[11]:


# do a permutation test to test for significance of difference

stars = df.stars.values
mean_diffs = []

for i in range(10000):
    perm = np.random.permutation(stars)
    mean_1 = np.mean(perm[:n_robinson])
    mean_2 = np.mean(perm[n_robinson:])
    mean_diffs.append(mean_1-mean_2)

plt.hist(mean_diffs);
plt.vlines(x=robinson_mean_diff, ymin = 0, ymax = 3000);
plt.xlabel('mean differences');
plt.ylabel('count');
plt.title('Distribution of Differences In Means');


# In[12]:


# calculate the p-value
p = sum(np.abs(mean_diffs) >= np.abs(robinson_mean_diff)) / len(mean_diffs)
print(p)


# **NOTE**: Getting rid of the records with lastname ROBINSON as the count of 5 stars from them seems suspiciously high relative to the rest of the sample.

# In[13]:


# drop rows with lastname ROBINSON
df = df[df.lastname != 'ROBINSON']


# In[14]:


# plot again and notice the change in distiribution
sns.catplot(x='stars', kind='count', data=df);
plt.title('Distribution of Stars Variable');


# In[15]:


# Reprinting .info(), note we still have missing values
df.info()


# ### Evaluate and Clean Price

# In[16]:


print(f'proportion of rows with missing price: {sum(df.price.isna()) / len(df):0.3f}')


# In[17]:


# before dealing with missing prices, check distribution
df.price.describe()


# In[18]:


sns.boxplot(df.price);
plt.title('Box Plot of Price Variable');


# In[19]:


# create new column encoding missing price
df['price_missing'] = df.price.isna()
print(f'proportion of price missing: {sum(df.price_missing)/len(df):0.3f}')


# In[20]:


# impute missing values with the mean
df.price.fillna(df.price.mean(),inplace=True)


# In[21]:


# check to make sure we're no longer missing data
assert sum(df.price.isna()) == 0


# In[22]:


# plot distribution of price, noting clusters. Depending on model, may want to bin?
sns.distplot(df.price, hist=False, rug=True);
plt.title('Distribution of Price Variable');


# ### Evaluate and Clean Favorite_Flower Variable

# In[23]:


# print out first few values for favorite_flower, noting that it is a categorical variable
df.favorite_flower.head()


# In[24]:


# print number of observations of each value
df.favorite_flower.value_counts()


# ## Engineer New Features

# In[25]:


# create new dataframe for engineered features
dfe = pd.DataFrame()


#  **NOTE**: our prediction is by family, need to collapse observations

# In[26]:


# collapse rows
g = df.groupby('lastname')


# ### Create Mean Price

# In[27]:


# get mean price per family
mean_prices = g.price.mean()
mean_prices.head()


# In[28]:


dfe['mean_price'] = mean_prices


# In[29]:


dfe.head()


# In[30]:


# depending on our model, may want to normalize features

dfe['mean_price_normed'] = (dfe.mean_price.values - dfe.mean_price.mean()) / dfe.mean_price.std()
dfe.mean_price_normed.agg(['mean','std']).round(2)


# In[31]:


dfe.head()


# In[32]:


# drop the unnormalized variable
dfe.drop('mean_price',axis=1,inplace=True)


# ## Create Median Stars

# **NOTE**: Using median to be robust against extreme high or low values

# In[33]:


g.stars.median().head()


# In[34]:


dfe['median_stars'] = g.stars.median()


# In[35]:


dfe.head()


# In[36]:


# again, depending on model, may or may not need to normalize
dfe['median_stars_normed'] = (dfe.median_stars.values / dfe.median_stars.mean()) / dfe.median_stars.std()
dfe.drop('median_stars',axis=1,inplace=True)


# In[37]:


dfe.head()


# ### Create Favorite Flower Dummies

# In[38]:


# transform favorite_flower into One Hot Encoding
flower_dummies = pd.get_dummies(df.favorite_flower, prefix='ff')
flower_dummies.head()


# In[39]:


# add the last name column using index
flower_dummies = df[['lastname']].join(flower_dummies)
flower_dummies.head()


# In[40]:


# group the flower columns by last name, aggregating by sum
flower_dummies = flower_dummies.groupby('lastname').sum()


# In[41]:


# simplify by transforming count of flower into 0,1
flower_dummies = flower_dummies.applymap(lambda x: int(x > 0))


# In[42]:


# join to our engineered values on lastname
dfe = dfe.join(flower_dummies)


# In[43]:


dfe.head()


# ## Create Labels

# In[44]:


# generate label for repeat customer
labels = df.lastname.value_counts().apply(lambda x: int(x > 1))


# In[45]:


labels.head()


# In[46]:


# join labels to engineered features
labels.name = 'reorder_label'

dfe = dfe.join(labels)

dfe.head()


# ## Train Classifier and Evaluate

# In[47]:


# get data and label column names
data_cols = dfe.columns[:-1]
label_col = dfe.columns[-1]


# In[48]:


data_cols


# In[49]:


label_col


# In[50]:


# importing here for demonstration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[51]:


# split into train and test (hold out) sets (using sklearn)
X_train, X_test, y_train, y_test = train_test_split(dfe.loc[:,data_cols],
                                                    dfe.loc[:,label_col],
                                                    test_size=0.2,
                                                    stratify=dfe.loc[:,label_col])
print(f'X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}')
print(f'X_test.shape = {X_test.shape}, y_train.shape = {y_test.shape}')


# In[52]:


# perform cross validation to tune parameters (not done here)
rf = RandomForestClassifier()
cv_scores = cross_val_score(rf, X_train, y_train, cv=3)
cv_scores


# In[53]:


print(f'mean cv accuracy: {np.mean(cv_scores):0.2f} +- {np.std(cv_scores)*2:0.2f}')


# In[54]:


# fit on training data and score on test
rf.fit(X_train,y_train)
print('test set accuracy: {:0.3f}'.format(rf.score(X_test,y_test)))


# ### Which Features are Most Important?

# In[55]:


for col,fi in sorted(list(zip(data_cols,rf.feature_importances_)),key=lambda x:x[1])[::-1]:
    print(f'{col:20s} : {fi:0.3f}')


# ### How well could we do just by guessing 1 for everyone?

# In[56]:


((dfe.loc[:,label_col] == 1).sum() / len(dfe)).round(3)

