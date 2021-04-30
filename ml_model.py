import numpy as np
import pandas as pd
import sklearn 

df = pd.read_csv("Dataset Diabetes Type1 (Total).csv")
df.head()

df = df.rename(columns={'Adequate Nutrition ':'Nutrition','Education of Mother':'MomEdu','Standardized growth-rate in infancy':'SGR','Standardized birth weight':'SBR','Impaired glucose metabolism ':'glu_met','Insulin taken':'Insulin','Family History affected in Type 1 Diabetes':'Type1','Family History affected in Type 2 Diabetes':'Type2','pancreatic disease affected in child ':'pancreatic'})

cage = pd.Series(df.Age).map({'greater then 15':'00','Less then 15':'01','Less then 11':'10','Less then 5':'11',})
csex = pd.Series(df.Sex).map({'Female':'0','Male':'1'})
cHb = pd.Series(df.HbA1c).map({'Over 7.5%':'1','Less then 7.5%':'0'})
cNutrition = pd.Series(df.Nutrition).map({'Yes':'1','No':'0'})
cMom = pd.Series(df.MomEdu).map({'Yes':'1','No':'0'})
canti =pd.Series(df.Autoantibodies).map({'Yes':'1','No':'0'}) 
cglu = pd.Series(df.glu_met).map({'Yes':'1','No':'0'})
cins = pd.Series(df.Insulin).map({'Yes':'1','No':'0'})
cD1 = pd.Series(df.Type1).map({'Yes':'1','No':'0'})
cD2 = pd.Series(df.Type2).map({'Yes':'1','No':'0'})
cHypo =  pd.Series(df.Hypoglycemis).map({'Yes':'1','No':'0'})
cHypo =  pd.Series(df.pancreatic).map({'Yes':'1','No':'0'})
cAff = pd.Series(df.Affected).map({'yes':'1','No':'0'})
cResidence = pd.Series(df['Area of Residence ']).map({'Suburban':'00','Urban':'01','Rural':'11'})


df2 = df.copy(deep=True)

df2['Age'] = cage
df2['Sex'] = csex
df2.HbA1c = cHb
df2.Nutrition = cNutrition
df2.MomEdu = cMom
df2.Autoantibodies = canti
df2.glu_met = cglu
df2.Insulin = cins
df2.Type1 = cD1
df2.Type2 = cD2
df2.Hypoglycemis = cHypo
df2.pancreatic = cHypo
df2.Affected = cAff
df2['Area of Residence '] = cResidence

df3 = df2.drop(['How Taken','SGR','SBR','Duration of disease','Other diease'],axis=1)

X = df3.iloc[:,:-1]
Y = df3.Affected

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_X = scaler.transform(X)


from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(scaled_X,Y)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(Xtrain,Ytrain)

import pickle
filename = 'model_diabetes.pkl'
pickle.dump(model, open(filename, 'wb'))

def Convert(data):
    ## here no need to encode
    ## transforming inputs using scaler
    y = scaler.transform(data)
    return y


## here we are pre-mapping the values so no encoding is required
