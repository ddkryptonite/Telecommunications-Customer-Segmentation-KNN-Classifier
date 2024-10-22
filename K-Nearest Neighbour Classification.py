#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()
df.describe()


# In[7]:


df.head(10)


# In[8]:


#Let’s see how many of each class is in our data set
df['custcat'].value_counts()


# In[11]:


df.hist(column='income', bins=50)


# In[3]:


#Let's define feature sets, X:

df.columns
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


# In[4]:


#Let's define feature sets, Y:

y = df['custcat'].values
y[0:5]


# In[5]:


#NORMALIZING DATA. Data Standardization gives the data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on the distance of data points:

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[17]:


#SPLITTING THE DATASEET INTO TRAIN AND TEST USING AN  IN-BUILT FUNCTION

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)#20% used for test and 80% used for training
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#APPLYING KNN CLASSIFIER

from sklearn.neighbors import KNeighborsClassifier

#TRAINING MODEL WITH TRAIN DATA. START WITH K = 4 FOR NOW

k = 9
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[18]:


#PREDICTING OUTCOME OF Y WHICH IS Y HAT

yhat = neigh.predict(X_test)
yhat[0:5]


# In[19]:


#EVALUATING FOR ACCURACY
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[26]:


#ITERATING THROUGH VARIOUS VALUES OF K STARTING FROM ONE TO FIND THE VALUE WHICH BEST PROVIDES THE BEST FIT

from sklearn.metrics import precision_score, mean_absolute_error

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
mae_scores = np.zeros((Ks-1))
precision_scores = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    # MAE
    mae_scores[n-1] = mean_absolute_error(y_test, yhat)

     # Precision
    precision_scores[n-1] = precision_score(y_test, yhat, average='micro')  # Use 'binary' for binary classification

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print("Mean Absolute Error Scores:", mae_scores)
print("Precision Scores:", precision_scores)

mean_acc


# In[20]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[ ]:




