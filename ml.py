#!/usr/bin/env python
# coding: utf-8

# In[151]:


get_ipython().system('pip install seaborn')
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings 

warnings.filterwarnings('ignore')


# In[152]:


df = pd.read_csv('insurance.csv')


# In[153]:


df


# # EDA  

# In[154]:


df.shape


# In[155]:


df.head()


# In[156]:


df.info()


# In[157]:


df.describe()


# In[158]:


df.isnull().sum()


# In[159]:


df.columns


# In[160]:


numeric_columns = ['age', 'bmi', 'children', 'charges']
for col in numeric_columns:
    plt.figure(figsize = (6,4))
    sns.histplot(df[col],kde = True,bins = 20)


# In[161]:


sns.countplot(x = df['children'])


# In[162]:


sns.countplot(x=df ['sex'])


# In[163]:


sns.countplot(x = df['smoker'])


# In[164]:


for columns in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[columns])
    plt.title(columns)
    plt.show()


# In[165]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only = True),annot = True)


# # Data Cleaning and Preprocessing 

# In[196]:


df_cleaned = df.copy()


# In[197]:


df_cleaned.head()


# In[168]:


df_cleaned.shape


# In[169]:


df_cleaned.drop_duplicates(inplace=True)


# In[170]:


df_cleaned.shape


# In[171]:


df_cleaned.isnull().sum()


# In[172]:


df_cleaned.dtypes


# In[173]:


df_cleaned['sex'].value_counts()


# In[174]:


df_cleaned['sex'] = df_cleaned['sex'].map({"male" : 0, "female" : 1})


# In[175]:


df_cleaned.head()


# In[176]:


df_cleaned['smoker'].value_counts()


# In[177]:


df_cleaned['smoker'] = df_cleaned['smoker'].map({"male" : 0, "female" : 1})


# In[178]:


df_cleaned


# In[179]:


df_cleaned.rename(columns = {
    'sex' : 'is_female',
    'smoker' : 'is_smoker',
}, inplace = True)


# In[180]:


df_cleaned.head()


# In[181]:


df['region'].value_counts()


# In[182]:


df_cleaned = pd.get_dummies(df_cleaned,columns = ['region'],drop_first = True)


# In[183]:


df_cleaned.head()


# In[184]:


bool_cols = df_cleaned.select_dtypes(include='bool').columns
df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)


# In[185]:


df_cleaned


# # Feature Engineering & Extraction

# In[186]:


sns.histplot(df['bmi'])


# In[187]:


df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']

)


# In[188]:


df_cleaned


# In[189]:


df_cleaned = pd.get_dummies(df_cleaned,columns = ['bmi_category'],drop_first = True)


# In[190]:


bool_cols = df_cleaned.select_dtypes(include='bool').columns
df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)


# In[191]:


df_cleaned.head()


# In[192]:


df_cleaned.columns


# In[193]:


from sklearn.preprocessing import StandardScaler

cols = ['age', 'bmi', 'children']

scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])


# In[201]:


df_cleaned.head()


# In[220]:


df_cleaned = df.copy()

# create target bin
df_cleaned['charged_bin'] = pd.qcut(df_cleaned['charges'], q=2, labels=[0,1])

# encode sex
df_cleaned['is_female'] = df_cleaned['sex'].map({'female':1, 'male':0})

# encode smoker
df_cleaned['is_smoker'] = df_cleaned['smoker'].map({'yes':1, 'no':0})

# encode region
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'])


# In[221]:


cat_features = [
    'is_female',
    'is_smoker',
    'region_northwest',
    'region_southeast',
    'region_southwest'
]

# keep only columns that actually exist
cat_features = [col for col in cat_features if col in df_cleaned.columns]
    


# In[222]:


results = []

for col in cat_features:
    
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charged_bin'])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        continue

    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    decision = "Reject Null (Keep Feature)" if p_val < 0.05 else "Accept Null (Drop Feature)"

    results.append([col, chi2_stat, p_val, decision])


chi2_df = pd.DataFrame(results, columns=['Feature', 'Chi2 Statistic', 'P-value', 'Decision'])

chi2_df = chi2_df.sort_values(by='P-value')

chi2_df


# In[223]:


from scipy.stats import chi2_contingency
import pandas as pd

for col in cat_features:
    
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charged_bin'])

    # skip bad tables
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        print(f"Skipping {col} (not enough variation)")
        continue

    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    print(f"{col}: p-value = {p_val}")
    


# In[224]:


results = []

for col in cat_features:
    
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charged_bin'])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        continue

    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    results.append((col, chi2_stat, p_val))


# In[225]:


chi_df = pd.DataFrame(results, columns=['Feature', 'Chi2 Stat', 'P-value'])

chi_df = chi_df.sort_values(by='P-value')

print(chi_df)


# In[228]:


df_cleaned.to_csv("final_insurance_project_df.csv", index=False)


# In[ ]:




