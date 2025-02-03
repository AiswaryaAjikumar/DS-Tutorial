import pandas as pd
import statsmodels.api as sm

file_path = "/mnt/data/Advertising.csv"
df = pd.read_csv(file_path)

# Drop the unnecessary index column
df = df.drop(columns=["Unnamed: 0"])

X = df[["TV", "radio", "newspaper"]]
y = df["sales"]

X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Extract required statistics
rse = model.mse_resid ** 0.
r_squared = model.rsquared  
f_statistic = model.fvalue  

print(f"Residual Standard Error (RSE): {rse}")
print(f"R-squared: {r_squared}")
print(f"F-statistic: {f_statistic}")

print(model.summary())
