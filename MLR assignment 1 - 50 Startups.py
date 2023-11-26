# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:34:36 2023

@author: sksha
"""
#Load the data
import numpy as np
import pandas as pd
df = pd.read_csv("50_Startups.csv")
df.dtypes
df.info()
df.head()
df.shape
list(df)

df = df.rename(columns={'R&D Spend':'RDS','Marketing Spend':'MKS'})
df
#======================================
# EDA
#EDA----->EXPLORATORY DATA ANALYSIS
import seaborn as sns
import matplotlib.pyplot as plt
data = ['RDS','Administration','MKS','Profit']
for column in data:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(f" Horizontal Box Plot of {column}")
    plt.show()
    
#There are no outliers present 
df.hist()
df.skew()
df.kurt()

#========================================================================
#Tranformation
#Continuous variable
df_cont = df[["RDS","Administration","MKS","Profit"]]
df_cont.shape
list(df_cont)

#Transformation on continous variables using Standardisation
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(df_cont)
SS_X = pd.DataFrame(SS_X)
SS_X.columns=list(df_cont)
SS_X

#Categorical variable 
df_cat =df.iloc[:,3:4]
df_cat.shape

#Transformation on Categorical variable using Label Encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Y = LE.fit_transform(df["State"])
Y = pd.DataFrame(Y)
Y.columns=list(df.iloc[:,3:4])
Y

df_final =pd.concat([SS_X,Y],axis=1)
df_final.info()

#==============================================================================
#Split the variables
X =df_final.drop(df_final.columns[[3]],axis=1)
X.info()
X

Y =df_final[df_final.columns[[3]]]
Y.info()
Y
#==============================================================================

""" FINDING THE CORRELATION BETWEEN ALL THE X VARIABLES AND TARGET VARIABLE (PROFIT)"""
df_final.corr()

# we have a larger relation with R&D Spend  with profit target variable so we will first fit that x variable 
# fit the linear regression model#
#==============================================================================
#Model fitting
#MODEL 1
Y = df_final["Profit"]
X = df_final[["RDS"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))                 #MSE = 0.053
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))  #RMSE = 0.231
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred) 
r2
print("r2:",r2.round(3))      #0.947%

#====================================================================================================
""" VIF (variance influence factor is one of the metric which is used to calculate the relationship between  the two independent variables in order to see
there is a presence of multi collinearity, if exists it will effect the accuracy score of the model. so the vif factor ranges follows as below mentioned
VIF = 1/1-r2 ,VIF < 5 no multi collinearity
VIF : 5- 10  some multi collinearity issues will be present but can be accepted
VIF > 10 not at all acceptable """

'''i will check collinearty between "X" features ,based on that i will build models to get best R square value''' 

#Checking collinearity between RDS,Administration
Y = df_final["RDS"]
X = df_final[["Administration"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)    #VIF =1.06218   # as VIF here between the mentioned variables is < 5 these can be taken together

#adding the administration column to the R_D_Spend 
#MODEL 2
Y = df_final["Profit"]    
X = df_final[["RDS","Administration"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))                #0.052
print(" Root Mean Squared Error:",np.sqrt(mse).round(3)) #0.228
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred) 
r2
print("r2:",r2.round(3))  #0.948

#==============================================================================
#Checking collinearity between RDS,Administration,State
Y = df_final["RDS"]
X = df_final[["State"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)    #1.0110804010836976  can be considered

Y = df_final["State"]
X = df_final[["Administration"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #1.0001403759001628  can be considered

#MODEL 3
Y = df_final["Profit"]     
X = df_final[["RDS","Administration","State"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))                #MSE = 0.052
print(" Root Mean Squared Error:",np.sqrt(mse).round(3)) #RMSE = 0.222
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred) 
r2
print("r2:",r2.round(3))      #0.948

"""here we are adding the x variables in order to increase the r2 value but the very next highly related variable is
Marketing Spend but before adding it we have to check for multi collinearity """
#==============================================================================
#Checking collinearity between RDS,MKS,Administration,State

Y = df_final["RDS"]
X = df_final[["MKS"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #2.103205816276043 can be considered

Y = df_final["Administration"]
X = df_final[["MKS"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #1.0010349416824806 can be considered

Y = df_final["State"]
X = df_final[["MKS"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #1.006069180313524  can be considered

#MODEL 4
Y = df_final["Profit"]     
X = df_final[["RDS","Administration","State","MKS"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))     #MSE = 0.049
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))  #RMSE = 0.222
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred) 
r2
print("r2:",r2.round(3))  #0.951

#from all the models, model 4 can be considered as best model it has less RMSE and high R square value 
#==============================================================================
"""by using stats models"""
import pandas as pd
import statsmodels.formula.api as smf
df = pd.read_csv("50_Startups.csv")
# Create an empty list to store R-squared values
rsquared_values = []

# Define the models and calculate R-squared for each
models = [
    'Profit~RDS',
    'Profit~RDS+Administration',
    'Profit~RDS+Administration+State',
    'Profit~RDS+Administration+State+MKS',
]

for model_formula in models:
    model = smf.ols(model_formula, data=df_final).fit()
    rsquared = model.rsquared
    rsquared_values.append(rsquared)

# Create a DataFrame to display the R-squared values
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
rsquared_df = pd.DataFrame({'Model': model_names, 'R-squared': rsquared_values})

print(rsquared_df) #Model 4 has high R square value = 0.95