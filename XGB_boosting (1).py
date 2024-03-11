#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib .pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#load dataset
data=pd.read_csv('heart.csv')


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


#univariate analysis
plt.figure(figsize=(20,25),facecolor='white')
plotnumber = 1

for column in data.columns:
    if plotnumber<=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.histplot(x=data[column])
        plt.xlabel(column, fontsize=20)
        
    plotnumber+=1
plt.show()


# In[9]:


#bivariate analysis

plt.figure(figsize=(20,25),facecolor='white')
plotnumber = 1

for column in data.columns:
    if plotnumber<=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.histplot(x=data[column])
        plt.xlabel(column, fontsize=20)
        plt.ylabel('HeartDisease',fontsize=20)
    plotnumber+=1
plt.show()


# In[10]:


#data preprocessing
#check null value
data.isnull().sum()


# In[11]:


#labelencoder to convert
from sklearn.preprocessing import LabelEncoder


# In[12]:


lb=LabelEncoder()
data.Sex=lb.fit_transform(data['Sex'])


# In[13]:


data.head()


# In[14]:


#chestpain type
data.head()


# In[15]:


data.rename({'ChestPainType':'cpt'},axis=1,inplace=True)


# In[ ]:





# In[16]:


data.cpt.unique()


# In[17]:


data.cpt.value_counts()


# In[18]:


data.loc[data['cpt']=='ASY','cpt']=3
data.loc[data['cpt']=='NAP','cpt']=2
data.loc[data['cpt']=='ATA','cpt']=1
data.loc[data['cpt']=='TA','cpt']=0





# In[19]:


data.head()


# In[20]:


data.RestingECG.unique()


# In[21]:


data.RestingECG.value_counts()


# In[22]:


data.loc[data['RestingECG']=='Normal','RestingECG']=2
data.loc[data['RestingECG']=='LVH','RestingECG']=1
data.loc[data['RestingECG']=='ST','RestingECG']=0




# In[23]:


data.head()


# In[24]:


#exercise angina
data.ExerciseAngina=lb.fit_transform(data['ExerciseAngina'])


# In[25]:


data.head()


# In[26]:


#ST slope
data.ST_Slope.unique()


# In[27]:


data.ST_Slope.value_counts()


# In[30]:


data.loc[data['ST_Slope']=='Flat','ST_Slope']=2
data.loc[data['ST_Slope']=='Up','ST_Slope']=1
data.loc[data['ST_Slope']=='Down','ST_Slope']=0



# In[31]:


data.head()


# In[33]:


data.describe()


# In[36]:


#feature selection
plt.figure(figsize=(30,30))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',annot_kws={'size':15})


# In[37]:


#model creation
X= data.drop('HeartDisease', axis=1)
Y=data.HeartDisease


# In[38]:


X


# In[39]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=3)


# In[40]:


#gradient boosting algo


# In[42]:


from sklearn .ensemble import GradientBoostingClassifier
gbm=GradientBoostingClassifier()#obj creation
gbm.fit(X_train,Y_train)#fitting data
Y_gbm=gbm.predict(X_test)#predicting 


# In[44]:


#evalution model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,classification_report
accu_scor=accuracy_score(Y_test,Y_gbm)
accu_scor


# In[47]:


get_ipython().system('pip install xgboost')


# In[48]:


import xgboost


# In[49]:


Y_train


# In[50]:


X_train.info()


# In[51]:


X_train.ST_Slope=X_train.ST_Slope.astype('int64')
X_train.RestingECG=X_train.RestingECG.astype('int64')
X_train.cpt=X_train.cpt.astype('int64')


# In[54]:


X_test.ST_Slope=X_test.ST_Slope.astype('int64')
X_test.RestingECG=X_test.RestingECG.astype('int64')
X_test.cpt=X_test.cpt.astype('int64')


# In[56]:


from xgboost import XGBClassifier
xgb_r=XGBClassifier()
xgb_r.fit(X_train,Y_train)
Y_hat=xgb_r.predict(X_test)


# In[57]:


print(classification_report(Y_test,Y_hat))


# In[59]:


from sklearn.model_selection import RandomizedSearchCV
6,
param_grid={'gamma':[0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
           'learning_rate':[0.01,0.03,0.06,0.1,0.15,0.2,0.25,0.300000012, 0.4, 0.5, 0.6, 0.7],
           'max_depth':[5,6,7,8,9,10,11,12,13,14],
           'n_estimators':[50,65,80,100,115,130,150],
           'reg_alpha':[0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
           'reg_lambda':[0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}

XGB=XGBClassifier(random_state=42,verbosity=0,silent=0)
rcv= RandomizedSearchCV(estimator=XGB, scoring='f1', param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)


# In[60]:


rcv.fit(X_train, Y_train)
cv_best_params=rcv.best_params_
print(f'Best paramters:{cv_best_params})')


# In[62]:


XGB2=XGBClassifier(reg_lambda= 51.2, reg_alpha= 0.1, n_estimators= 130, max_depth= 9, learning_rate= 0.1, gamma=0)
XGB2.fit(X_train, Y_train)
Y_predict=XGB2.predict(X_test)
f1_score=f1_score(Y_predict,Y_test)


# In[63]:


f1_score


# In[ ]:




