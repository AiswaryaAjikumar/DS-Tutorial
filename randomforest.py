import numpy as np  # for numerical computations
import pandas as pd  # for data manipulation
from sklearn.model_selection import train_test_split  # function to split dataset
from sklearn.ensemble import RandomForestRegressor  # Import RandomForest model
from sklearn.metrics import mean_squared_error  # function to evaluate model performance

# Load dataset
file_path = "Advertising.csv"  # file path of the dataset
df = pd.read_csv(file_path)  # Read the dataset into a pandas DataFrame


X = df[['TV', 'radio', 'newspaper']].values  # Select features
y = df['sales'].values  # Select target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

n_estimators = 5  # Number of individual Random Forest models
predictions = []  # List to store predictions from each model

for i in range(n_estimators):  # Loop through the number of models
    model = RandomForestRegressor(n_estimators=10, random_state=i)  # Initialize a Random Forest with 10 trees
    model.fit(X_train, y_train)  # Train the model on training data
    preds = model.predict(X_test)  # Predict target values for test data
    predictions.append(preds)  # Append predictions to list


final_predictions = np.mean(predictions, axis=0)  # Compute the average of predictions from all models


mse = mean_squared_error(y_test, final_predictions)  # Compute Mean Squared Error
print(f"Mean Squared Error: {mse}")  # Print the error metric
