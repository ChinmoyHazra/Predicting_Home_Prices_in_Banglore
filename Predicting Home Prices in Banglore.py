#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[3]:


df1= pd.read_csv("E:/Python Project/Bengaluru_House_Data.csv")


# In[4]:


df1.head()


# In[5]:


df1.shape


# In[7]:


df1.columns


# In[8]:


df1['area_type'].unique()


# In[9]:


df1['area_type'].value_counts()


# In[10]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.shape


# In[11]:


df2.isnull().sum()


# In[12]:


df3=df2.dropna()


# In[13]:


df3.isnull().sum()


# In[14]:


df3.shape


# In[15]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[18]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[22]:


df4 = df3[~df3['total_sqft'].apply(is_float)]


# In[24]:


df4['total_sqft'].unique()


# In[25]:


import re
def convert_sqft_to_num(x):
    tokens= x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        head = re.split("[^0-9]",x)
        tail = "".join(re.split("[^a-zA-Z]",x))


        if tail == "Perch":
            if head[1] != "":
                return round(((float(head[0]) * 272.25) + (float(head[1]) * 2.7225)),2)
            else:
                return round((float(head[0]) * 272.25),2)
        if tail == "SqMeter":
            if head[1]!="":
               return round(((float(head[0])*10.7639104)+(float(head[1])*0.107639104)),2)
            else:
                return round((float(head[0])*10.7639104),2)
        if tail == "SqYards":
            if head[1]!="":
               return round(((float(head[0])*9)+(float(head[1])*0.09)),2)
            else:
                return round((float(head[0])*9),2)

        if tail == "Guntha":
            if head[1]!="":
               return round(((float(head[0])*1089)+(float(head[1])*10.89)),2)
            else:
                return round((float(head[0])*1089),2)
        if tail == "Acres":
            if head[1]!="":
               return round(((float(head[0])*43560)+(float(head[1])*435.60)),2)
            else:
                return round((float(head[0])*43560),2)

        if tail == "Cents":
            if head[1]!="":
               return round(((float(head[0])*435.6)+(float(head[1])*4.3560)),2)
            else:
                return round((float(head[0])*435.60),2)

        if tail == "Cents":
            if head[1] != "":
                return round(((float(head[0]) * 435.6) + (float(head[1]) * 4.3560)), 2)
            else:
                return round((float(head[0]) * 435.60), 2)

        if tail == "":
            return float(head[0])
    except:
        return None


# In[26]:


df3['total_sqft']= df3['total_sqft'].apply(convert_sqft_to_num)


# In[27]:


df3['total_sqft'].unique()


# In[28]:


df5 = df3.copy()


# In[30]:


df5['price_per_sqft']=(df5['price']*100000)/df5['total_sqft']


# In[31]:


df5.head()


# In[32]:


df5_stats = df5['price_per_sqft'].describe()
df5_stats


# In[33]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[34]:


location_stats.values.sum()


# In[36]:


len(location_stats[location_stats>10])


# In[37]:


len(location_stats)


# In[38]:


len(location_stats[location_stats<=10])


# In[40]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[41]:


len(df5.location.unique())


# In[42]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[43]:


df5.head(10)


# In[45]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[46]:


df5.shape


# In[47]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[49]:


df6.price_per_sqft.describe()


# In[50]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[51]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[52]:


plot_scatter_chart(df7,"Hebbal")


# In[53]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[54]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[56]:


plot_scatter_chart(df8,"Hebbal")


# In[57]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[58]:


df8.bath.unique()


# In[59]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[60]:


df8[df8.bath>10]


# In[61]:


df8[df8.bath>df8.bhk+2]


# In[62]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[64]:


df9.head(2)


# In[65]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# In[66]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[67]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[69]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[71]:


df12.shape


# In[72]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[73]:


X.shape


# In[75]:


y = df12.price
y.head(3)


# In[76]:


len(y)


# In[90]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[91]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[86]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[88]:


predict_price('1st Phase JP Nagar',1000, 2, 2)

