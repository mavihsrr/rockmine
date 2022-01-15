#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[47]:


sonar_data = pd.read_csv(r"C:\Users\shiva\Desktop\Copy of sonar data.csv", header= None)


# In[48]:


sonar_data.head()


# In[50]:


#number of columns and rows
sonar_data.shape


# In[52]:


sonar_data.describe()


# In[55]:


sonar_data[60].value_counts()


# In[56]:


sonar_data.groupby(60).mean()


# In[60]:


#seperating data and labels
X = sonar_data.drop(columns = 60 ,axis =1)
Y = sonar_data[60]


# In[62]:


print(X)
print(Y)


# Training and Test data 

# In[65]:


X_train , X_test  , Y_train , Y_test = train_test_split(X,Y, test_size = 0.1 , stratify = Y , random_state=1)


# In[78]:


print(X.shape , X_test.shape , X_train.shape)
print(X_train)
print(Y_train)


# MODEL TRAINING
# 

# In[79]:


model = LogisticRegression()


# In[80]:


#training the logistic regression model with training data
model.fit(X_train, Y_train)


# model evaluation

# In[83]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_prediction = accuracy_score(X_train_prediction, Y_train)


# In[88]:


print("Accuracy on training data : ", training_data_prediction)


# In[90]:


#accuracy on test DATA
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[91]:


print("Accuracy on test data is - " , test_data_accuracy)


# MAKING A PREDICTIVE SYSTEM

# In[102]:


input_data = (0.0164,0.0173,0.0347,0.0070,0.0187,0.0671,0.1056,0.0697,0.0962,0.0251,0.0801,0.1056,0.1266,0.0890,0.0198,0.1133,0.2826,0.3234,0.3238,0.4333,0.6068,0.7652,0.9203,0.9719,0.9207,0.7545,0.8289,0.8907,0.7309,0.6896,0.5829,0.4935,0.3101,0.0306,0.0244,0.1108,0.1594,0.1371,0.0696,0.0452,0.0620,0.1421,0.1597,0.1384,0.0372,0.0688,0.0867,0.0513,0.0092,0.0198,0.0118,0.0090,0.0223,0.0179,0.0084,0.0068,0.0032,0.0035,0.0056,0.0040)

#changing input_data to a numpy array 
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we're predictiing for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction == 'R'):
    print("The object is a rock")
else:    
    print("The object is a mine")


# In[ ]:





# In[ ]:





# In[ ]:




