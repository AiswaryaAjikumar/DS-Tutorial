import numpy as np  # Import NumPy for numerical computations
import pandas as pd  # Import pandas for data manipulation
from sklearn.model_selection import train_test_split  # Import function to split dataset
from sklearn.metrics import mean_squared_error  # Import function to evaluate model performance
from scipy.cluster.hierarchy import linkage, dendrogram  # Import hierarchical clustering functions
import matplotlib.pyplot as plt  # Import Matplotlib for visualization
from mlxtend.frequent_patterns import apriori, association_rules  # Import Apriori algorithm functions

# Sample dataset for Apriori algorithm (list of transactions)
transactions = [
    ['milk', 'bread', 'nuts', 'apple'], 
    ['milk', 'bread', 'nuts'],  
    ['milk', 'bread'], 
    ['milk', 'bread', 'apple'],  
    ['bread', 'nuts', 'apple'], 
    ['bread', 'apple'],  
    ['milk', 'bread', 'nuts', 'apple'],  
    ['milk', 'apple']  
]

# Convert transactions into a DataFrame format
items = sorted(set(item for sublist in transactions for item in sublist))

# Create a binary representation of transactions (1 if item is in transaction, else 0)
transaction_df = pd.DataFrame([{item: (item in transaction) for item in items} for transaction in transactions])

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(transaction_df, min_support=0.3, use_colnames=True)

# Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0) 

# Print the frequent itemsets found by the algorithm
print("Frequent Itemsets:")
print(frequent_itemsets)

# Print the association rules derived from the frequent itemsets
print("\nAssociation Rules:")
print(rules)
