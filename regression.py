
import pandas as pd  # importing pandas
import statsmodels.api as sm  # For logistic regression
from sklearn.model_selection import train_test_split  # For splitting data 
from sklearn.preprocessing import StandardScaler  # importing sklearn
from sklearn.metrics import accuracy_score, classification_report  # For evaluating model performance


data = pd.read_csv('membership.csv')  # Read the dataset 

X = data[['Age', 'Income']]  # Features
y = data['Buys Membership']  # Target variable

# Add a constant column for predictor
X = sm.add_constant(X) 

# Split the dataset into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  


# Initialize StandardScaler for feature scaling
scaler = StandardScaler()

# Create copies of training and testing 
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Standardize the 'Age' and 'Income'
X_train_scaled[['Age', 'Income']] = scaler.fit_transform(X_train[['Age', 'Income']])  
X_test_scaled[['Age', 'Income']] = scaler.transform(X_test[['Age', 'Income']])  


model = sm.Logit(y_train, X_train_scaled)  # Creating a logistic regression model
result = model.fit()  # Fitting the model 
pred_probs = result.predict(X_test_scaled)  # Predict probabilities of buying membership
y_pred = (pred_probs >= 0.5).astype(int)  # Convert probabilities to binary 


accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy score
report = classification_report(y_test, y_pred)  # Generate a classification report (precision, recall, F1-score)


print("Model Accuracy:", accuracy)  # Print accuracy
print("Classification Report:\n", report)  # Print classification metrics
print(result.summary())  # Print detailed logistic regression model summary

# New data for prediction
new_data = pd.DataFrame({
    'const': 1,  # Add constant term for prediction 
    'Age': [28, 50],  # New inputs for Age
    'Income': [40000, 85000]  # New inputs for Income
})

# Standardize the new input data using the same scaler
new_data[['Age', 'Income']] = scaler.transform(new_data[['Age', 'Income']])

# Predict probabilities for the new data
new_predictions = result.predict(new_data)

# Convert probabilities to binary class labels
final_predictions = (new_predictions >= 0.5).astype(int)

# Print the predictions for the new data points
print("Predictions for new inputs:", final_predictions.values)  
# Output will be 0 or 1 showing whether the person is predicted to buy the membership or not

