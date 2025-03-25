import numpy as np  #  for numerical computations
import pandas as pd  # for data manipulation
from sklearn.model_selection import train_test_split  # to split dataset
from sklearn.metrics import mean_squared_error  # to evaluate model performance
from scipy.cluster.hierarchy import linkage, dendrogram  #hierarchical clustering functions
import matplotlib.pyplot as plt  # Import Matplotlib for visualization


file_path = "Advertising.csv"  # Define the file path of the dataset
df = pd.read_csv(file_path)  # Read the dataset into a pandas

# Selecting features for clustering
X = df[['TV', 'radio', 'newspaper']].values  # Convert independent variables to NumPy array

linkage_matrix = linkage(X, method='complete')  # Compute linkage matrix using complete linkage

# Plot the dendrogram
plt.figure(figsize=(10, 5))  # Set figure size
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")  # Set title
plt.xlabel("Data Points")  # Set x-axis label
plt.ylabel("Distance")  # Set y-axis label
dendrogram(linkage_matrix)  # Generate and plot dendrogram
plt.show()  # Display the plot
