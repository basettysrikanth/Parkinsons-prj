#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


import pandas as pd
dataset = pd.read_csv("parkinsons main.csv")


# In[5]:


dataset.head()


# In[6]:


dataset.tail()


# In[7]:


print(dataset.shape)
dataset.dtypes


# In[8]:


dataset.describe()


# In[9]:


import seaborn as sb
corr_map=dataset.corr()
sb.heatmap(corr_map,square=True)


# In[10]:


import matplotlib.pyplot as plt
import numpy as np

# K value means how many features required to see in heat map
k=10

# finding the columns which related to output attribute and we are arranging from top coefficient correlation value to downwards.
cols=corr_map.nlargest(k,'status')['status'].index

# correlation coefficient values
coff_values=np.corrcoef(dataset[cols].values.T)
sb.set(font_scale=1.25)
sb.heatmap(coff_values,cbar=True,annot=True,square=True,fmt='.2f',
           annot_kws={'size': 10},yticklabels=cols.values,xticklabels=cols.values)
plt.show()


# In[11]:


correlation_values=dataset.corr()['status']
correlation_values.abs().sort_values(ascending=False)


# In[12]:


dataset.info()


# In[13]:


dataset.isna().sum()


# In[14]:


# split the dataset into input and output attribute.

y=dataset['status']
cols=['MDVP:RAP','Jitter:DDP','DFA','NHR','MDVP:Fhi(Hz)','name','status']
x=dataset.drop(cols,axis=1)


# In[15]:


train_size=0.80
test_size=0.20
seed=5

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train_size,test_size=test_size,random_state=seed)


# In[16]:


n_neighbors=5
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# keeping all models in one list
models=[]
models.append(('LogisticRegression',LogisticRegression()))
models.append(('knn',KNeighborsClassifier(n_neighbors=n_neighbors)))
models.append(("decision_tree",DecisionTreeClassifier()))
models.append(("XGBoost",XGBClassifier()))

# Evaluating Each model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
names=[]
predictions=[]
error='accuracy'
for name,model in models:
    fold=KFold(n_splits=10,random_state=0)
    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    

# Visualizing the Model accuracy
fig=plt.figure()
fig.suptitle("Comparing Algorithms")
plt.boxplot(predictions)
plt.show()


# In[47]:


import seaborn as sns
path = 'parkinsons.data'
df = pd.read_csv("parkinsons main.csv")
plt.figure(figsize = (15,10))
sns.pairplot(df, vars=['MDVP:Fo(Hz)','MDVP:Flo(Hz)','HNR','PPE','spread1','spread2'],hue='status',palette='Dark2')
plt.savefig('Relationship')
plt.show()


# In[48]:


X = np.array(df.drop(['name','status'], axis = 1))
y = np.array(df['status'])
print(f'X shape: {X.shape} Y Shape: {y.shape}')


# In[51]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)
def crossValidate(model):
    #Using StratifiedKFold to ensure that the divided folds are shuffled
    strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
    
    #Getting just specific scores for perfromance evualation.
    scoring = ["accuracy","precision","recall","f1","roc_auc"]
    cv = cross_validate(model, scaled_X, y, cv = strat_k_fold, scoring = scoring)
    result = [round(cv[score].mean(),3) for score in cv]
    return result
model = XGBClassifier()


# In[57]:


result[2:]


# In[58]:


plt.figure(figsize = (9,5))
model_preformance = pd.Series(data=result[5:], 
        index=['Accuracy','Precision','Recall','F1-Score','AUC (ROC)'])
model_preformance.sort_values().plot.barh()
plt.title('Model Performance')


# In[ ]:




