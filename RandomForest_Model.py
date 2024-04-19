#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv('Dataset1.csv')


# In[4]:


df.head


# In[5]:


print(df.shape)


# In[6]:


x = df.iloc[:,:7].to_numpy()
y = df.iloc[:,7:].to_numpy()


# In[7]:


from sklearn.preprocessing import StandardScaler

# assume X is your data matrix with shape (n_samples, n_features)
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df)
print(df_normalized)


# In[8]:


#X_1=df.iloc[:,:7].to_numpy()
#Y_1=df.iloc[:,7:].to_numpy()


# In[9]:


df_normalized


# In[10]:


print(df_normalized.shape)


# In[11]:


#X= df_normalized[:, :7]
#Y= df_normalized[:, 7:]
#print(X.shape)
#print(Y.shape)


# In[12]:


from sklearn.decomposition import PCA


# In[23]:


#trying out the random forest
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[24]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42)
rf.fit(X_train, y_train)


# In[26]:


y_pred = rf.predict(X_test)


# In[27]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[28]:


print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Accuracy:", r2)


# In[29]:


X_test[:5,:]


# In[37]:


a = np.max(np.abs(y_test), axis=1)>=X_test[:,-1] # These case where the magnetorquer reaches saturation


# In[35]:


np.max(np.abs(y_test), axis=1)


# In[39]:


a = np.max(np.abs(y), axis=1)>=x[:,6]


# In[41]:


dot_product = [np.dot(val1, val2) for val1, val2 in zip(y,x[:,3:6])]


# In[44]:


np.max(np.abs(dot_product))


# In[22]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [5,10,15, 20],
    'n_estimators': [50,100,200,500]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, Y)
print("Best parameters: ", grid_search.best_params_)
print("Best negative mean squared error: ", grid_search.best_score_)


# In[ ]:





# In[ ]:





# In[21]:


y1_train = y_train[:,0]  # first column of y
y2_train = y_train[:,1]  # second column of y
y3_train = y_train[:,2] 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
svr1 = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr2 = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr3 = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr1.fit(X_train, y1_train)
svr2.fit(X_train, y2_train)
svr3.fit(X_train, y3_train)


# In[ ]:


y1_pred = svr1.predict(X_test)
y2_pred = svr2.predict(X_test)
y3_pred = svr3.predict(X_test)


# In[ ]:


y_pred = np.column_stack((y1_pred, y2_pred, y3_pred))


# In[ ]:


r2 = r2_score(y_test, y_pred)
print(r2)

