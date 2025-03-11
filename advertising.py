import pandas as pd  # Importing pandas 
import statsmodels.api as sm  # Importing statsmodels

file_path = "Advertising.csv"  #file path for the dataset
df = pd.read_csv(file_path)  # Reading the CSV file 


df = df.drop(columns=["Unnamed: 0"])  # Removing an unnamed index column if it exists


X = df[["TV", "radio", "newspaper"]]  #features
y = df["sales"]  # target 

X = sm.add_constant(X)  # Adding a constant term for the regression model


model = sm.OLS(y, X).fit()  # Performing regression and fitting the model

rse = model.mse_resid ** 0.5  # Calculating Residual Standard Error 
r_squared = model.rsquared  # calculating R-squared value
f_statistic = model.fvalue  #calculating  F-statistic


print(f"Residual Standard Error (RSE): {rse}")  # printing RSE to measure model accuracy
print(f"R-squared: {r_squared}")  # printing R-squared 
print(f"F-statistic: {f_statistic}")  #printing F-statistic 


print(model.summary())  #printing detailed regression summary
