import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('/mnt/data/membership.csv')
X = data[['Age', 'Income']]
y = data['Buys Membership']
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[['Age', 'Income']] = scaler.fit_transform(X_train[['Age', 'Income']])
X_test_scaled[['Age', 'Income']] = scaler.transform(X_test[['Age', 'Income']])

model = sm.Logit(y_train, X_train_scaled)
result = model.fit()

pred_probs = result.predict(X_test_scaled)
y_pred = (pred_probs >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)
print(result.summary())

new_data = pd.DataFrame({
    'const': 1,  # Adding constant for prediction
    'Age': [28, 50],
    'Income': [40000, 85000]
})
new_data[['Age', 'Income']] = scaler.transform(new_data[['Age', 'Income']])
new_predictions = result.predict(new_data)
final_predictions = (new_predictions >= 0.5).astype(int)

print("Predictions for new inputs:", final_predictions.values)
