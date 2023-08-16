#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imblearn
from sklearn.impute import KNNImputer
import category_encoders as ce
from sklearn.model_selection import cross_val_score,cross_validate
from imblearn.over_sampling import SMOTE
import catboost
import lightgbm
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,f1_score,recall_score,precision_score,confusion_matrix,accuracy_score,roc_auc_score,roc_curve


# In[4]:


#loading dataset
data=pd.read_csv("C:/Users/sande/OneDrive/Desktop/project_data.csv")


# In[5]:


data.head()


# In[6]:


data=data.sample(frac=1,random_state=10)


# In[7]:


data


# In[8]:


data.shape


# In[ ]:





# # EDA

# In[9]:


data.info()


# In[46]:


data['gender'].value_counts()
male_candidate=6701
female_candidate=554


# In[14]:


#percentage of female candidate contests election
data['gender'].value_counts()
print('percentage of female candidate:',round((554/6701)*100,2),'%')


# In[113]:


data['seat_category'].value_counts()


# In[114]:


#total number of candidate caste wise
data['caste'].value_counts()


# In[20]:


# name of all party who participated in election
data['party'].unique()


# In[21]:


# total number of unique party who contests election
len(data['party'].unique())


# In[116]:


#number of seats on which party contest election 
data['party'].value_counts()


# In[34]:


#average age of candidate
print('average age of candidate:',data['age'].mean())

#average age of candidate who won the election
print('average age winning candidate:',data[data['result']==1]['age'].mean())


# In[36]:


data


# In[55]:


#number of male winning candidate 
male_winner=(data[data['gender']=='MALE']['result'].sum())
print(male_winner)

#number of female winning candidate
female_winner=(data[data['gender']=='FEMALE']['result'].sum())
print(female_winner)


# In[59]:


#strike rate gender wise
male_str_rate=(male_winner/male_candidate)*100
print('male strike rate:',male_str_rate)

female_str_rate=(female_winner/female_candidate)*100
print('female strike rate:',female_str_rate)


# # Graphical Analysis

# In[117]:


#correlation matrix 
plt.figure(figsize=[15,10])
sns.heatmap(data.corr(),annot=True)


# In[35]:


#Histogram for age column
print('average age of candidate:',round(data['age'].mean()))
sns.histplot(data['age'],bins=10)


# In[31]:


#average age of candidate who won the election
data[data['result']==1]['age'].mean()


# In[119]:


# Check basic statistics for categorical columns

categorical_columns = ['gender', 'seat_category', 'caste', 'party','result']

# Create subplots for count plots
fig, axes = plt.subplots(len(categorical_columns), 1, figsize=(8, 20))

# Plot the count plots
for i, column in enumerate(categorical_columns):
    sns.countplot(x=data[column], ax=axes[i])
    axes[i].set_title(f'Count of each category in {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()


# In[120]:


#Clustered Bar Chart:
data.groupby(['gender', 'result']).size().unstack().plot(kind='bar', stacked=True)
plt.xlabel('gender')
plt.ylabel('Count')
plt.title('Clustered Bar Chart: Count of result by gender')
plt.show()


# # DATA PROCESSING

# In[121]:


#droping year and candidate
data=data.drop(columns=['year','candidate'],axis=1)


# In[122]:


#mapping gender column
data['gender']=data['gender'].map({'MALE':1,'FEMALE':0,'THIRD':0})


# In[123]:


#creating dummies variable for seat_category
data=pd.get_dummies(data,columns=['seat_category'])
data=data.drop(['seat_category_ST'],axis=1)


# In[124]:


#filling missing value with obc
#data['caste']=data['caste'].fillna('obc')

#creating dummies variable for caste
data=pd.get_dummies(data,columns=['caste'])
data=data.drop(['caste_st'],axis=1)


# In[125]:


#target encoding

#encoder=ce.TargetEncoder(cols=['party'])
#encoder_party=encoder.fit_transform(data['party'],data['result'])
#data['encoder_party']=encoder_party

data["party_encoded"] = data.groupby("party")["result"].transform("mean")+0.0001
data=data.drop(['party'],axis=1)


# In[126]:


data.duplicated().value_counts()


# In[127]:


data.drop_duplicates(inplace=True)


# In[181]:


#pairplot
plt.figure(figsize=[15,10])
sns.pairplot(data)


# In[183]:


data


# In[194]:


#distribution plot
plt.figure(figsize=(15,12))

plt_cols = ['gender','age','result','contested','mus_pop','seat_category_GEN','party_encoded','seat_category_SC','caste_gen','caste_obc','caste_sc']

# I didnt plot the star_f columns because they are highly correlated to norating1 and noreviews1 
# columns & have similar distributions to them

k=1
for i in plt_cols:
    plt.subplot(4,3,k)
    sns.distplot(data[i])
    k=k+1


# In[154]:


#dividing input and output
x=data.drop(['result'],axis=1)
y=data['result']


# # Model Building

# In[155]:


#Here im using imblearn pipeline insted of pipeline because it handle sampling properly
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# In[156]:


std=StandardScaler()
x_train_std=std.fit_transform(x)


# In[157]:


#lOGISTIC REGESSION
lr=LogisticRegression()
resampling=SMOTE()
pipeline=Pipeline([('SMOTE',resampling),('LOGISTIC' ,lr)])
scores=cross_val_score(pipeline,x_train_std,y,scoring='f1',cv=5)
scores


# In[158]:


print('Average Score is:',scores.mean())
print('Standard Deviation of score:',np.std(scores))


# In[159]:


#LightGBM
lgb=LGBMClassifier()
resampling=SMOTE()
pipeline=Pipeline([('SMOTE',resampling),('LightGBM' ,lgb)])
scores=cross_val_score(pipeline,x_train_std,y,scoring='f1',cv=5)
scores


# In[160]:


print('Average Score is:',scores.mean())
print('Standard Deviation of score:',np.std(scores))


# In[161]:


#XGBOOST CLASSIFIER
xgb=XGBClassifier()
resampling=SMOTE()
pipeline=Pipeline([('SMOTE',resampling),('logistic regression' ,xgb)])
scores=cross_val_score(pipeline,x_train_std,y,scoring='f1',cv=5)
scores


# In[162]:


print('Average Score is:',scores.mean())
print('Standard Deviation of score:',np.std(scores))


# In[163]:


#Decision Tree
dt=DecisionTreeClassifier()
resampling=SMOTE()
pipeline=Pipeline([('SMOTE',resampling),('Decision Tree' ,dt)])
scores=cross_val_score(pipeline,x_train_std,y,scoring='f1',cv=5)
scores


# In[164]:


print('Average Score is:',scores.mean())
print('Standard Deviation of score:',np.std(scores))


# In[165]:


#Random Forest
rf=RandomForestClassifier()
resampling=SMOTE()
pipeline=Pipeline([('SMOTE',resampling),('Random Forest' ,rf)])
scores=cross_val_score(pipeline,x_train_std,y,scoring='f1',cv=5)
scores


# In[166]:


print('Average Score is:',scores.mean())
print('Standard Deviation of score:',np.std(scores))


# In[171]:


model=VotingClassifier(estimators=[('lr',lr),('dt',dt),('lgb',lgb),('xgb',xgb),('rf',rf)],voting='soft')
resampling=SMOTE()
pipeline=Pipeline([('SMOTE',resampling),('Random Forest' ,model)])
scores=cross_val_score(pipeline,x_train_std,y,scoring='f1',cv=5)
scores


# In[172]:


print('Average Score is:',scores.mean())
print('Standard Deviation of score:',np.std(scores))


# # Model Building using train_test_split

# In[173]:


#train_test_split
from sklearn.model_selection import train_test_split


# In[174]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[175]:


sm = SMOTE(random_state=40)
x_res, y_res = sm.fit_resample(x_train,y_train)


# In[176]:


std=StandardScaler()
x_train_std=std.fit_transform(x_res)
x_test_std=std.transform(x_test)


# In[177]:


lr=LogisticRegression()
lr.fit(x_train_std,y_res)
y_pred=lr.predict(x_test_std)
f1_score(y_test,y_pred)


# In[178]:


confusion_matrix(y_test,y_pred)


# In[179]:


lgb=LGBMClassifier()
lgb.fit(x_train_std,y_res)
y_pred=lgb.predict(x_test_std)
f1_score(y_test,y_pred)


# In[180]:


confusion_matrix(y_test,y_pred)


# In[ ]:




