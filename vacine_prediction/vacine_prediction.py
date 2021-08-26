#Projet : COVID

#I.Defined objectif : 
    
#objectif : Epidemic data to vacination prediction

#metric : ROC AUC for each of the two target variables. The mean of these two scores will be the overall score. A higher value indicates stronger performance. 

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
#II.EDA(Exploraty Data Analysis)

#load data
training_set_features = pd.read_csv('training_set_features.csv')
training_set_labels = pd.read_csv('training_set_labels.csv')

test_set_features = pd.read_csv('test_set_features.csv')

#1.shape analysis

#explore first fives rows ==> df values
training_set_features.head()
training_set_labels.head()

#target identification : h1n1_vaccine and seasonal_vaccine

#shape : features(26707,36) and labels(26707,3)
training_set_features.shape 
training_set_labels.shape     

#variable types (23 float64) (12 object) (1 int64) and labels(3 int)
training_set_features.dtypes.value_counts()
training_set_labels.dtypes.value_counts()

#Nan values : health_insurance,employment_industry,employment_occupation = 50% Nan values / income_poverty = 15% Nan values / Other = <1% Nan values and No Nan values in labels
sns.heatmap(training_set_features.isna(), cbar=False)

(training_set_features.isna().sum()/training_set_features.shape[0]).sort_values(ascending = True)

sns.heatmap(training_set_labels.isna(), cbar=False)

#2.data analysis

#drop columns unusable
training_set_features = training_set_features.drop(['respondent_id','health_insurance', 'employment_industry', 'employment_occupation'], axis=1)
training_set_labels = training_set_labels.drop('respondent_id', axis=1)


#RELATION : VARIABLE DISTRIBUTIONS

#target vizualisation : h1n1_vacine = 21% of positive values and 79% of negative values and seasonal_vaccine : 47% of positive values and 53% of negative values
training_set_labels['h1n1_vaccine'].value_counts(normalize = True)
training_set_labels['seasonal_vaccine'].value_counts(normalize = True)

#Signification des variables
#continue variables histograms : standardisées, skewed (asymétrique), test sanguin
for col in training_set_features.select_dtypes('float'):
    plt.figure()
    sns.distplot(training_set_features[col].dropna())

    
#qualitative variables : race: max white and 75000_above poverty and rent
for col in training_set_features.select_dtypes('object'):
    plt.figure()
    training_set_features[col].value_counts().plot.pie()       
       
#RELATION : TARGET-VARIABLE

training_set= pd.concat([training_set_features, training_set_labels], axis=1, sort=False)
    
#Created positive and negative under set of 'h1n1_vaccine'
h1n1_vaccine_positive = training_set[training_set['h1n1_vaccine'] == 1]   
h1n1_vaccine_negative = training_set[training_set['h1n1_vaccine'] == 0] 

#Created positive and negative under set of 'seasonal_vaccine'
seasonal_vaccine_positive = training_set[training_set['seasonal_vaccine'] == 1]   
seasonal_vaccine_negative = training_set[training_set['seasonal_vaccine'] == 0] 
    
#Created opinion and behavioral set
opinion_columns = training_set.columns[training_set.columns.str.startswith('opinion')]
behavioral_columns = training_set.columns[training_set.columns.str.startswith('behavioral')]

#Taget 'h1n1_vaccine'/opinion : h1n1_vacc_effective,h1n1_risk, seas_vacc_effective,seas_risk
for col in opinion_columns:
    plt.figure()
    sns.distplot(h1n1_vaccine_positive[col].dropna(), label='positive')
    sns.distplot(h1n1_vaccine_negative[col].dropna(), label = 'negative')
    plt.legend()

#Target 'h1n1_vaccine'/viral : behavioral_avoidance, behavioral_wash_hands,behavioral_wash_gatherings, behavioral_outside_home,behavioral_touch_face  
for col in behavioral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(training_set['h1n1_vaccine'],training_set[col]), annot=True,fmt='d')

#Taget 'seasonal_vaccine'/opinion : h1n1_vacc_effective, h1n1_risk, seas_vacc_effective,seas_risk
for col in opinion_columns:
    plt.figure()
    sns.distplot(seasonal_vaccine_positive[col].dropna(), label='positive')
    sns.distplot(seasonal_vaccine_negative[col].dropna(), label = 'negative')
    plt.legend()

#Target 'seasonal_vaccine'/viral : 
for col in behavioral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(training_set['seasonal_vaccine'],training_set[col]), annot=True,fmt='d')
    
    
#Target 'seasonal_vaccine' : 62% no h1n1 vacine and 38% h1n1 vacine
seasonal_vaccine_positive['h1n1_vaccine'].value_counts(normalize = True)
#Target 'seasonal_vaccine' : 93% no h1n1 vacine and 7% h1n1 vacine
seasonal_vaccine_negative['h1n1_vaccine'].value_counts(normalize = True)

#Target 'h1n1_vaccine' : 17% no h1n1 vacine and 83% h1n1 vacine
h1n1_vaccine_positive['seasonal_vaccine'].value_counts(normalize = True)

#Target 'h1n1_vaccine' : 63% no h1n1 vacine and 37% h1n1 vacine
h1n1_vaccine_negative['seasonal_vaccine'].value_counts(normalize = True)
    
    
#Variable/variable relations

#opinion/opinion : correlation between h1n1 and seas 
sns.clustermap(training_set[opinion_columns].corr())

#behavioral/behavioral :
sns.clustermap(training_set[behavioral_columns].corr())      

#NULL HYPOTHESIS (H0):

#Les individus non vacinés ont des opinions significativement différents
#H0 = les opinions sont égaux chez les individus vacinés contre h1n1 et non vacinés contre h1n1.

##import package 
from scipy.stats import ttest_ind

h1n1_vaccine_positive.shape
h1n1_vaccine_negative.shape

balanced_neg = h1n1_vaccine_negative.sample(h1n1_vaccine_positive.shape[0])

#t-test
def t_test(col):
    alpha =0.02
    stat, p = ttest_ind(balanced_neg[col].dropna(), h1n1_vaccine_positive[col].dropna())
    if p< alpha:
        return 'HO rejetée'
    else: 
        return 0

for col in opinion_columns :
    print (f'{col :-<50} {t_test(col)}')
    
for col in behavioral_columns :
    print (f'{col :-<50} {t_test(col)}')
    
#Les individus non vacinés ont des opinions significativement différents
    

#H0 = les opinions sont égaux chez les individus vacinés contre seas et non vacinés contre seas.
seasonal_vaccine_positive.shape
seasonal_vaccine_negative.shape

balanced_neg = seasonal_vaccine_negative.sample(seasonal_vaccine_positive.shape[0])

#t-test
def t_test(col):
    alpha =0.02
    stat, p = ttest_ind(balanced_neg[col].dropna(), seasonal_vaccine_positive[col].dropna())
    if p< alpha:
        return 'HO rejetée'
    else: 
        return 0

for col in opinion_columns :
    print (f'{col :-<50} {t_test(col)}')

for col in behavioral_columns :
    print (f'{col :-<50} {t_test(col)}')
        
  
#III.Pre-processing
                
#Split Train and Test set       
from sklearn.model_selection import train_test_split 
trainset, testset = train_test_split(training_set, test_size=0.2, random_state=0)
   
#    trainset['h1n1_vaccine'].value_counts()
#    testset['h1n1_vaccine'].value_counts()
#    
#    trainset['seasonal_vaccine'].value_counts()
#    testset['seasonal_vaccine'].value_counts() 
#    
#   
#    def value_counts(df):
#        
#        columns = []
#        for col in trainset.select_dtypes('object') :
#            columns.append(col)
#        
#        values = []
#        for col in columns:
#            val = df[col].value_counts()
#            val.name = col
#            values.append(val)
#    
#        df_value_counts = pd.concat(values, axis=1)
#        
#        return df_value_counts
#
#trainset_value_counts= value_counts(trainset)
#testset_value_counts = value_counts(testset)    
#
#
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer

def encodage(df):
    df = pd.get_dummies(df)
    return df
       
def imputation(df):
    return df.dropna(axis=0)
 
def preprocessing(df):
    
    df = imputation(df)
    df = encodage(df)
    
     
    X = df.drop(['h1n1_vaccine','seasonal_vaccine'], axis =1)
    y = df['h1n1_vaccine']
    
    X2 = df.drop(['h1n1_vaccine','seasonal_vaccine'], axis =1)
    y2 = df['seasonal_vaccine']
    
    print(y.value_counts(), y2.value_counts())
    
    return X,y,X2,y2

X_train, y_train, X2_train, y2_train = preprocessing(trainset)
X_test, y_test, X2_test, y2_test  = preprocessing(testset)

X_train.shape
y_train.shape
X2_train.shape
y2_train.shape


X_test.shape
y_test.shape
X2_test.shape
y2_test.shape

#
#def col_lack (df1, df2):       
#  return list(set(df1.columns) - set(df2.columns))
#
#col_lack (X_train, X_test)
#
#X_test['employment_industry_qnlwzans'] = X_test.apply(lambda x: 0, axis=1)
#X2_test['employment_industry_qnlwzans'] = X2_test.apply(lambda x: 0, axis=1)
          
        
#4.Modelling  
from sklearn.svm import SVC

model = SVC(probability=True)
model2 = SVC(probability=True)
#5.Procédure d'évaluation
from sklearn.metrics import roc_auc_score

def evaluation(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test.values, y_pred, average="macro")
    
    return roc_auc

    
evaluation(model, X_train, y_train, X_test, y_test)
 
evaluation(model2, X2_train, y2_train, X2_test, y2_test)  

#6. Prediction

test_set_features = test_set_features.drop(['respondent_id','health_insurance', 'employment_industry', 'employment_occupation'], axis=1)

#Preprocessing
def preprocessing_pred(df):       
    X = encodage(df)            
    return X
    
X_pred = preprocessing_pred(test_set_features) 
 
#prediction set'shape 
X_pred.shape

#Nan values in prediction set
sns.heatmap(test_set_features.isna(), cbar=False)   
(X_pred.isna().sum()/X_pred.shape[0]).sort_values(ascending = True)
X_pred = X_pred.replace(np.nan, 0)


#Predictict probability h1n1_vaccine & seasonal_vaccine
h1n1_vaccine = model.predict_proba(X_pred)[:,1]
seasonal_vaccine = model2.predict_proba(X_pred)[:,1]


#Import data in submission_format
submission_format = pd.read_csv('submission_format.csv')

submission_format['h1n1_vaccine'] = h1n1_vaccine
submission_format['seasonal_vaccine'] = seasonal_vaccine

#Distribution h1n1_vaccine & seasonal_vaccine values
submission_format['h1n1_vaccine'].hist()
submission_format['seasonal_vaccine'].hist()
