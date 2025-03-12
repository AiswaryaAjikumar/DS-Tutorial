import numpy as np #importing numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #importing sklearn


X = np.array([[2, 3], [5, 6]])  # Two data points with two features
y = np.array([0, 1])  # Corresponding class labels

#using library
# Initialize LDA model 
lda = LinearDiscriminantAnalysis(n_components=1)

# Fit LDA model
X_lda_sklearn = lda.fit_transform(X, y)

# Print LDA transformed data from scikit-learn
print("LDA Transformed Data (sklearn):\n", X_lda_sklearn)



#using normal matrix multiplication
# Compute class means (mean vector of each class)
mean_0 = X[y == 0].mean(axis=0)  # Mean of class 0
mean_1 = X[y == 1].mean(axis=0)  # Mean of class 1 

# Compute within-class scatter matrix 
S_W = np.cov(X.T)  # Transpose X to get feature covariance

# Compute between-class scatter matrix
mean_diff = (mean_1 - mean_0).reshape(-1, 1)  # Compute mean difference as a column vector
S_B = np.dot(mean_diff, mean_diff.T)  # Outer product to form S_B

# Solve the eigenvalue problem 
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Choose the eigenvector with the largest eigenvalue 
lda_direction = eigvecs[:, np.argmax(eigvals)]

# Print the LDA transformation direction 
print("LDA Direction Vector (manual computation):\n", lda_direction)

