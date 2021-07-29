# import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from pandas_profiling import ProfileReport 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif # use this for classification tasks
from sklearn.ensemble import RandomForestClassifier

# load the data
df = pd.read_csv('kidney_disease.csv')
print(df.head(10))
print(df.shape)
print(df.columns.values)

df.drop('id', axis=1, inplace=True)
print(df.info())
print(df.head().T)

#Encoding categorical variables
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'normal':0,'abnormal':1})
df[['ba','pcc']] = df[['ba','pcc']].replace(to_replace={'notpresent':0,'present':1})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df["classification"] = [1 if i == "ckd" else 0 for i in df["classification"]]

print(df.head().T)

print(df.dtypes)

df.pcv = pd.to_numeric(df.pcv, errors='coerce')
df.pc = pd.to_numeric(df.pc, errors='coerce')
df.dm = pd.to_numeric(df.dm, errors='coerce')
df.cad = pd.to_numeric(df.cad, errors='coerce')
df.wc = pd.to_numeric(df.wc, errors='coerce')
df.rc = pd.to_numeric(df.rc, errors='coerce')

#describe data 
print(df.describe().T)

print(sum(df.duplicated()))

df.isna().sum().sort_values()

((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False).plot(kind='bar', figsize=(10,10))
plt.show()

#show missing data
import missingno as msno

msno.matrix(df)
plt.show()


#Data visualization
plt.style.use("seaborn-dark-palette")
sns.countplot(df.classification)
plt.xlabel('Chronic Kidney Disease')
plt.title("patients Classification",fontsize=15)
plt.show()

# blood pressure graph
sns.factorplot(data=df, x='bp', kind= 'count',size=6,aspect=2)
plt.xlabel('Chronic Kidney Disease')
plt.title("blood pressure graph",fontsize=15)
plt.show()

#density-frequency graph

sns.factorplot(data=df, x='sg', kind= 'count',size=6,aspect=2)
plt.xlabel('Chronic Kidney Disease')
plt.title("density-frequency graph",fontsize=15)
plt.show()

#sugar-frequency graph
sns.factorplot(data=df, x='su', kind= 'count',size=6,aspect=2)
plt.xlabel('Chronic Kidney Disease')
plt.title("sugar-frequency graph",fontsize=15)
plt.show()

df.age.value_counts().sort_values()


# packed cell volume grahp
sns.factorplot(data=df, x='age', kind= 'count',aspect=5)
plt.xlabel('Chronic Kidney Disease')
plt.title("packed cell volume grahp",fontsize=15)
plt.show()

#sns.pairplot(df )
#plt.show()

#correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.title('Correlations between different predictors')
plt.show()

df2 = df.dropna(axis = 0)
print(f"Before dropping all NaN values: {df.shape}")
print(f"After dropping all NaN values: {df2.shape}")

X = df2.drop(['classification', 'sg', 'appet', 'rc', 'pcv', 'hemo', 'sod'], axis = 1)
y = df2['classification']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Import Libraries
from sklearn.ensemble import RandomForestClassifier
#----------------------------------------------------

#----------------------------------------------------
#Applying RandomForestClassifier Model 

'''
ensemble.RandomForestClassifier(n_estimators='warn’, criterion=’gini’, max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                                max_features='auto’,max_leaf_nodes=None,min_impurity_decrease=0.0,
                                min_impurity_split=None, bootstrap=True,oob_score=False, n_jobs=None,
                                random_state=None, verbose=0,warm_start=False, class_weight=None)
'''

RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=20,max_depth=2,random_state=33) #criterion can be also : entropy 
RandomForestClassifierModel.fit(X_train, y_train)

#Calculating Details
print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))
print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))


#calculating prediction
y_pred = RandomForestClassifierModel.predict(X_test)
y_pred_prob = RandomForestClassifierModel.predict_proba(X_test)
print('Predicted Value for RandomForestClassifierModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for RandomForestClassifierModel is : ' , y_pred_prob[:10])

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=RandomForestClassifierModel.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("GBCModel Matrix")
plt.show()

# Saving the model
import pickle
pickle.dump(RandomForestClassifierModel, open('kidney.pkl', 'wb'))