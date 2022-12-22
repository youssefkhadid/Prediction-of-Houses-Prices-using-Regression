#!/usr/bin/env python
# coding: utf-8

# In[1]:


# major libraries
import pandas as pd
import os

#sklearn libraries for pipepline preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


# In[2]:


#reading the CSV file using Pandas
file_path = os.path.join(os.getcwd(), 'housing.csv')
df_housing = pd.read_csv(file_path)

# change <1H OCEAN to 1H OCEAN because it will give us errors later
df_housing['ocean_proximity'] = df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN')

# Feature Engineering => Feature extraction => Add new columns to the main DataFrame
df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedrooms_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
df_housing['population_per_household'] = df_housing['population'] / df_housing['households']

# we split the dataset to target and features
x = df_housing.drop(columns='median_house_value', axis=1)
y= df_housing['median_house_value']
# random split of dataset to two sets(train_set, test_set)
# for validation we'll use cross validation
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, test_size=0.15, random_state=42)

# seperating the columns according to their type (numerical or categorical)
num_cols = [col for col in x_train.columns if x_train[col].dtype in ['float32', 'float64', 'int32', 'int64']]
categ_cols = [col for col in x_train.columns if x_train[col].dtype not in ['float32', 'float64', 'int32', 'int64']]


# In[3]:


num_Pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(num_cols)),
    ('Imputer', SimpleImputer(strategy='median')),
               ('Scaler', StandardScaler())
               ])

categ_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(categ_cols)),
    ('Imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(sparse=False))
])

total_pipeline = FeatureUnion(transformer_list=[
    ('num', num_Pipeline),
    ('categ', categ_pipeline)
])

# we deal with the total_pipeline as an instance, we fit and transform the train set and transform only for test set
X_train_final = total_pipeline.fit_transform(x_train)


# In[4]:


def preprocess_new(X_new):
    ''' This function tries to process the new instances before prediction using the Model
    Args:
    *****
       (X_new, 2D array) ==> the Features in the same order
               ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population'
               'households', 'median_income', 'ocean_proximity']
               all the Features are numerical, except the last one is categorical
    Returns:
    *******
           Preprocessed Features ready to make inference by the model
    '''
    return total_pipeline.transform(X_new)

