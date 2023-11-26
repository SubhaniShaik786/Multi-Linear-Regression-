# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:02:42 2023

@author: sksha
"""

#importing the data
import numpy as np
import pandas as pd
df = pd.read_csv("ToyotaCorolla.csv",encoding = 'latin')
df.head()

df = df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
df.info()
df.dtypes
df.shape

# EDA #
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
df = df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
for column in df:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
#so basically we have seen the ouliers for each variable using seaborn#

"""removing the ouliers"""
# List of column names with continuous variables
df = df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for df.cloumns in df:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker_Length = Q1 - 1.5 * IQR
    upper_whisker_Length = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker_Length) & (data_without_outliers[column]<= upper_whisker_Length)]

# Print the cleaned data without outliers
print(data_without_outliers)
df = data_without_outliers
print(df)
df.shape
df.info() 

#Histogram,skewness , KURTOSIS
df.hist()
df.skew()
df.kurt()
df[df.duplicated()]
df = df.drop_duplicates().reset_index(drop=True)
df.describe()
df

#correlation analysis
df.corr()

#Continous variables
df_cont = df.iloc[:,1:9]
df_cont.info()

#Standardisation
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(df_cont)
SS_X
X = pd.DataFrame(SS_X)
X.columns = list(df_cont)
X

# Y variable
Y_trans = df.iloc[:,0:1]
Y_trans
list(Y_trans)
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_Y = SS.fit_transform(Y_trans)
SS_Y
Y = pd.DataFrame(SS_Y)
Y.columns = list(Y_trans)
Y

#final transformed data
df_final = pd.concat([X,Y],axis = 1)
df_final

#Data Visualisation
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)
sns

# Pairplot to visualize the relationships between variables
import matplotlib.pyplot as plt
sns.set_style(style='darkgrid')
sns.pairplot(df_final, vars=["Age_08_04", "KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"])
plt.show()

# Correlation heatmap
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

#correlation
pd.set_option('display.max_columns', None)	
df_final.corr()	

# in multilinear regression we check every X variable's relation with the Y variable 
# --> here we keep on adding each x variable to our model one by one so then we can descide which model is best
# x variables = Age_08_04,KM,HP,cc,Doors,Gears,Quarterly_Tax,Weight
# Y variable = Price
#======================================================================================================
# Model 1
Y = df_final["Price"]
X = df_final[["Age_08_04"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y) #b0+b1x1
LR.intercept_ #b0
LR.coef_ #b1
#predicted values
Y_pred = LR.predict(X)
Y_pred
#calculating sum of errors 
from sklearn.metrics import mean_squared_error
import numpy as np
error = mean_squared_error(Y, Y_pred)
print("MSE :",error.round(3))           #MSE : 0.226
print("RMSE :",np.sqrt(error).round(3)) #RMSE : 0.476
#r^2 error
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)
print("R square :",(r2*100).round(3))
# RMSE : 0.476 , R square : 77.355

#======================================================================================================
""" checking for multi collineaity """
 
Y = df_final["Age_08_04"]
X = df_final[["KM"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.3364447147688487

Y = df_final["Age_08_04"]
X = df_final[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #Variance Influence Factor:  1.1793165628383648

Y = df_final["KM"]
X = df_final[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #Variance Influence Factor:  1.0045541797596278

# Model 2
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))   #0.835

#======================================================================================================
Y = df_final["HP"]
X = df_final[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0000805312722458

Y = df_final["KM"]
X = df_final[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.1246382490498912

Y = df_final["Age_08_04"]
X = df_final[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0106168073460788


# Model 3
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))  #r2: 0.841


Y = df_final["Quarterly_Tax"]
X = df_final[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.100567159032836

Y = df_final["Quarterly_Tax"]
X = df_final[["KM"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0872702145535558

Y = df_final["Quarterly_Tax"]
X = df_final[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.3846437869675032

Y = df_final["Quarterly_Tax"]
X = df_final[["Age_08_04"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.038823083668639

#MODEL 4
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP","Quarterly_Tax"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))           #Mean Squared Error: 0.139
print(" Root Mean Squared Error:",np.sqrt(mse).round(3)) # Root Mean Squared Error: 0.372
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))  #r2: 0.841


Y = df_final["Doors"]
X = df_final[["KM"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #Variance Influence Factor:  1.0012400900150542

Y = df_final["Doors"]
X = df_final[["Age_08_04"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0220644031397967

Y = df_final["Doors"]
X = df_final[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.1001621861776982

Y = df_final["Doors"]
X = df_final[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.0084993700235525

Y = df_final["Doors"]
X = df_final[["Quarterly_Tax"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0116591090343787

#MODEL 5
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))  #r2: 0.841

Y = df_final["cc"]
X = df_final[["Age_08_04"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0094093679344913

Y = df_final["Doors"]
X = df_final[["cc"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.0063208614219397

Y = df_final["cc"]
X = df_final[["Quarterly_Tax"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.1032960832309824

Y = df_final["cc"]
X = df_final[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.001241044401372

Y = df_final["cc"]
X = df_final[["KM"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.0108965313107474

Y = df_final["cc"]
X = df_final[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.126476637315796

#MODEL 6
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors","cc"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))  #r2: 0.85

""" as the Gears column consists of less correlation with all other independent variables multi collinearity doesnt exists
between any of them so we can consider Gears into model fitting"""

#MODEL 7
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors","cc","Gears"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))                #Mean Squared Error: 0.137
print(" Root Mean Squared Error:",np.sqrt(mse).round(3)) # Root Mean Squared Error: 0.371
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))  #r2: 0.851

#This above model is the best model for the given dataset with low rmse as 0.371 and R2 score of 85%#

#==================================================================================
# Residual Analysis
#fit the model with seaborn,statsmodels package
import pandas as pd
df_residual = pd.read_csv("ToyotaCorolla.csv",encoding="latin")
df_residual

#format the plot background as scatter plots for all variables
import seaborn as sns
sns.set_style(style="darkgrid")
sns.pairplot(df_final)

#build a model
import statsmodels.formula.api as smf
model = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=df_final).fit()
model.summary()


import matplotlib.pyplot as plt
import statsmodels.api as sm

qqplot = sm.qqplot(model.resid,line = "q")
plt.title("Normal Q-Q plot of residuals")
plt.show()

import numpy as np
list(np.where((model.resid) > 10))


rsquared_values=[]
# Define the models and calculate R-squared for each
models = [
    'Price~Age_08_04',
    'Price~Age_08_04+KM',
    'Price~Age_08_04+KM+Weight',
    'Price~Age_08_04+KM+Weight+HP',
    'Price~Age_08_04+KM+Weight+HP+Quarterly_Tax',
    'Price~Age_08_04+KM+Weight+HP+Quarterly_Tax+Doors',
    'Price~Age_08_04+KM+Weight+HP+Quarterly_Tax+Doors+cc',
    'Price~Age_08_04+KM+Weight+HP+Quarterly_Tax+Doors+cc+Gears',
]
import statsmodels.formula.api as smf
for model_formula in models:
    model = smf.ols(model_formula, data=df_final).fit()
    rsquared = model.rsquared
    rsquared_values.append(rsquared)

# Create a DataFrame to display the R-squared values
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4','Model 5', 'Model 6', 'Model 7', 'Model 8']
rsquared_df = pd.DataFrame({'Model': model_names, 'R-squared': rsquared_values})

print(rsquared_df)

#model 8 has best R-squared value = 0.850697