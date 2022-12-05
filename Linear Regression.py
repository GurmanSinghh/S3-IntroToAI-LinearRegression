#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.feature_selection import RFE
import os

#Loading data (Ecom dataframe)
path = "/Users/gurman/Desktop"
filename = 'Ecom Expense.csv'
fullpath = os.path.join(path,filename)

#Storing the data into a variable
ecom_exp_gurman = pd.read_csv(fullpath)


#Initial exploration
ecom_exp_gurman.head(3)
ecom_exp_gurman.tail()
ecom_exp_gurman.shape
ecom_exp_gurman.columns
ecom_exp_gurman.dtypes
ecom_exp_gurman.isna().sum()
ecom_exp_gurman.info()


#Removing 'Transaction ID' column for it is only a sequential index. Therefore, of little value for our analysis.
ecom_exp_gurman.drop('Transaction ID', axis=1, inplace=True)

#Confirming the removal of the Transaction ID Column
ecom_exp_gurman.head()

#Confirming the number of categories in Gender and City Tier 
ecom_exp_gurman.Gender.value_counts()
ecom_exp_gurman["City Tier"].value_counts()

#Transforming categorical variables into numerical variables
gender_dummies = pd.get_dummies(ecom_exp_gurman.Gender, prefix="Sex")
cityTier_dummies= pd.get_dummies(ecom_exp_gurman["City Tier"], prefix="City")

#Concatenating the newly created dummy variables to the base dataframe
df_ecom = pd.concat([ecom_exp_gurman, gender_dummies, cityTier_dummies], axis = 1)

#Check accuracy of concatenation
df_ecom.columns

#Drop original categorical variables
df_ecom.drop(["Gender", "City Tier"], axis = 1, inplace=True)

df_ecom.head()

#Function to normalize data
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df_ecom.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return df_norm

#Applying normalization on the dataset
df_ecom_norm = min_max_scaling(df_ecom)

pd.set_option('display.max_columns', None)

#Check normalization
df_ecom_norm.head(2)

df_ecom_norm.columns

#Plot histograms for all variables to check the data distribution
df_ecom_norm.hist(figsize=(9,10), bins=(30))

#Plot scatter matrix
pd.plotting.scatter_matrix(df_ecom_norm[['Age ', 'Monthly Income', 'Transaction Time', 'Total Spend']], figsize=[13, 15],hist_kwds={'bins':30}, alpha=0.4, diagonal='kde')

#Renaming columns to remove unnecessary spaces
df_ecom_norm.columns
df_ecom_norm = df_ecom_norm.rename(columns={'Age ': 'Age', ' Items ':'Items', 'Monthly Income': 'Monthly_Income', 'Transaction Time':'Transaction_Time', 'Total Spend':'Total_Spend', 'City_Tier 1': 'City_Tier_1', 'City_Tier 2':'City_Tier_2', 'City_Tier 3':'City_Tier_3'})
df_ecom_norm.columns

df_ecom_norm.columns

#Selecting predictors to feed the model
df_x = df_ecom_norm[['Monthly_Income', 'Transaction_Time', 'Sex_Female', 'Sex_Male', 'City_Tier_1', 'City_Tier_2', 'City_Tier_3']]


df_x.head(3)

#Target variable
df_y = df_ecom_norm['Total_Spend']

df_y.head(3)

#Splitting the dataset into training and test data
X_train_gurman, X_test_gurman, y_train_gurman, y_test_gurman = train_test_split(df_x, df_y, test_size=0.35, random_state=86)

#Instantiating a Linear model
model = LinearRegression()

#Fitting the model onto the training data
model.fit(X_train_gurman, y_train_gurman)

#Intercept and coeficients
print(model.intercept_)
print(model.coef_)

#Calculating the model R^2 based on the test data
model.score(X_test_gurman, y_test_gurman)

#Predicting the target variable by using the test predictors
y_prediction = model.predict(X_test_gurman)

#Inserting another feature (Record) to the list of predictors
df_x2 = df_ecom_norm[['Monthly_Income', 'Transaction_Time', 'Record', 'Sex_Female', 'Sex_Male', 'City_Tier_1', 'City_Tier_2', 'City_Tier_3']]

#Splitting the new dataset
X_train_gurman2, X_test_gurman2, y_train_gurman2, y_test_gurman2 = train_test_split(df_x2, df_y, test_size=0.35, random_state=86)

#Fitting the model onto the new data
model.fit(X_train_gurman2, y_train_gurman2)

#Intercept and coeficients
print(model.intercept_)
print(model.coef_)

#R^2
model.score(X_test_gurman2, y_test_gurman2)

#y_prediction2 = 


#Using statsmodels to get a model summary with and without the Record variable
import statsmodels.formula.api as sm

df_train = pd.concat([X_train_gurman, y_train_gurman], axis = 1)

df_train.head()
df_train.columns

model2=sm.ols(formula='Total_Spend~Monthly_Income+Transaction_Time+Sex_Female+Sex_Male+City_Tier_1+City_Tier_2+City_Tier_3',data=df_train).fit()

print(model2.summary())

df_train_record = pd.concat([X_train_gurman2, y_train_gurman2], axis = 1)
df_train_record.head()
df_train_record.columns

model3 = sm.ols(formula='Total_Spend~Monthly_Income+Transaction_Time+Record+Sex_Female+Sex_Male+City_Tier_1+City_Tier_2+City_Tier_3',data=df_train_record).fit()

print(model3.summary())


# In[ ]:




