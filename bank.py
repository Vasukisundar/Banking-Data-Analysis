#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().system('pip install SQLAlchemy pymysql')


# In[12]:


get_ipython().system('pip install pymysql')


# In[2]:


from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


engine = create_engine("mysql+pymysql://root:root@127.0.0.1:3306/banking_case")

query = "SELECT * FROM customer"
df = pd.read_sql(query, engine)

print(df.head())


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


bins = [0,100000,300000,float('inf')]
lables = ['low','medium','high']
df['Income Band'] = pd.cut(df['Estimated Income'],bins=bins,labels=lables,right='false')


# In[11]:


df['Income Band'].value_counts().plot(kind='bar',xlabel='Income bands')


# In[12]:


categorical_cols = df[["BRId", "GenderId", "IAId", "Amount of Credit Cards", "Nationality", "Occupation", "Fee Structure", "Loyalty Classification", "Properties Owned", "Risk Weighting", "Income Band"]].columns

for col in categorical_cols:
    print(f"Value Counts for:'{col}':")
    display(df[col].value_counts())


# In[13]:


#Categorical Univariate Analysis

for i, predictor in enumerate(df[["BRId", "GenderId", "IAId", "Amount of Credit Cards", "Nationality", "Occupation", "Fee Structure", "Loyalty Classification", "Properties Owned", "Risk Weighting", "Income Band"]].columns):
    plt.figure(i)
    sns.countplot(data=df,x=predictor)


# In[14]:


#Categorical Bivariate Analysis

for i,predictor in enumerate(df[["BRId", "GenderId", "IAId", "Amount of Credit Cards", "Nationality", "Occupation", "Fee Structure", "Loyalty Classification", "Properties Owned", "Risk Weighting", "Income Band"]].columns):
    plt.figure(i)
    sns.countplot(data=df,x=predictor,hue="Nationality")


# In[25]:


#Numerical Analysis
numerical_cols = ['Estimated Income', 'Superannuation Savings', 'Credit Card Balance', 'Bank Loans', 'Bank Deposits', 'Checking Accounts', 'Saving Accounts', 'Foreign Currency Account', 'Business Lending']
plt.figure(figsize=(15,13),layout='constrained')
for i,col in enumerate(numerical_cols):
    plt.subplot(4,3,i+1)
    sns.histplot(df[col],kde=True)
    plt.title(col)
plt.show()


# In[23]:


#Numerical Correlation Analysis

numerical_cols = ['Estimated Income', 'Superannuation Savings', 'Credit Card Balance', 'Bank Loans', 'Bank Deposits', 'Checking Accounts', 'Saving Accounts', 'Foreign Currency Account', 'Business Lending']
correlation_matrix=df[numerical_cols].corr()

plt.figure(figsize=(6,6))
sns.heatmap(correlation_matrix, annot=True, cmap='crest', fmt=".2f")
plt.show


# In[41]:


#Numerical Bivariate Analysis

pairs_to_plot = [
    ('Bank Deposits', 'Saving Accounts'),
    ('Checking Accounts', 'Saving Accounts'),
    ('Checking Accounts', 'Foreign Currency Account'),
    ('Age', 'Superannuation Savings'),
    ('Estimated Income', 'Checking Accounts'),
    ('Bank Loans', 'Credit Card Balance'),
    ('Business Lending', 'Bank Loans'),
]

for x_cols,y_cols in pairs_to_plot:
    plt.figure(figsize=(8,8))
    sns.regplot(data=df,
                x=x_cols,
                y=y_cols,
                scatter_kws={'alpha': 0.4},     # semi-transparent points
                line_kws={'color': 'red'}       # best-fit line color
               )

plt.title(f'Relationship between {x_cols} and {y_cols}', fontsize=10)
plt.xlabel(x_cols, fontsize=6)
plt.ylabel(y_cols,fontsize=6)
plt.tight_layout()
plt.show()


# ##Deposits and Savings Behavior
# 
# The high correlation between Bank Deposits and Saving Accounts suggests that these may either measure overlapping financial behavior (e.g., total funds a customer keeps in the bank) or that people who actively deposit funds also tend to maintain or grow savings balances.
# 
# 
# ## Income, Age, and Accumulation
# 
# Moderate correlations of Age and Estimated Income with various balances (Superannuation, Savings, Checking) reflect a common financial lifecycle trend: higher income earners and older individuals often accumulate more savings, retirement funds, and may carry higher credit card balances or loans.
# 
# 
# ##Low Correlation with Properties Owned
# 
# Property ownership may depend on external factors (location, real estate market conditions, inheritance, etc.) that are not captured by these particular banking variables. Hence, we see weaker correlations here.
# 
# 
# ##Business vs. Personal Banking
# 
# 
# Business Lending’s moderate link to Bank Loans suggests some customers may have both personal and business debts. However, business lending is relatively uncorrelated with other deposit or property-related metrics, indicating it may serve a distinct subset of customers or needs.
# 

# In[ ]:




