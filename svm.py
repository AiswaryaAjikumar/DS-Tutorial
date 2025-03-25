import numpy as np  # for numerical computations
import pandas as pd  # for data manipulation
from sklearn.model_selection import train_test_split  # to split dataset
from sklearn.metrics import mean_squared_error  # to evaluate model performance

# Load dataset
file_path = "Advertising.csv"  # Define the file path of the dataset
df = pd.read_csv(file_path)  # Read the dataset into a pandas DataFrame

# Selecting features and target
X = df[['TV', 'radio', 'newspaper']].values  # Convert independent variables to NumPy array
y = df['sales'].values  # Convert dependent variable to NumPy array

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Initialize parameters
learning_rate = 0.001  # Learning rate 
lambda_param = 0.01  # Regularization parameter
n_iters = 1000  # Number of iterations for training
n_samples, n_features = X_train.shape  # Get number of samples and features
y_train_binary = np.where(y_train <= np.median(y_train), -1, 1)  #  binary labels : -1 for low, 1 for high
w = np.zeros(n_features)  # Initialize weights to zero
b = 0  # Initialize bias to zero


for _ in range(n_iters):  # Iterate for specified number of times
    for idx, x_i in enumerate(X_train):  # Iterate over each training sample
        condition = y_train_binary[idx] * (np.dot(x_i, w) - b) >= 1  # Check if the data point is correctly classified
        if condition:  # If correctly classified, update weights only
            w -= learning_rate * (2 * lambda_param * w)  # Update weights using regularization term
        else:  # If misclassified, update both weights and bias
            w -= learning_rate * (2 * lambda_param * w - np.dot(x_i, y_train_binary[idx]))  # Update weights
            b -= learning_rate * y_train_binary[idx]  # Update bias

# Prediction function
def predict(X):
    approx = np.dot(X, w) - b  # Compute decision function
    return np.where(approx >= 0, 1, -1)  # Assign class label based on decision function

# Predict labels for test data
y_pred = predict(X_test)

# Convert predicted labels back to original range
mse = mean_squared_error(np.where(y_pred == -1, np.min(y_train), np.max(y_train)), y_test)  # Compute mean squared error between predictions and actual values
print(f"Mean Squared Error using SVM: {mse}")  # Print the error metric
