from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn import tree
import graphviz

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor model
dt_regressor = DecisionTreeRegressor(random_state=42)

# Train the model
dt_regressor.fit(X_train, y_train)

# Make predictions
y_pred_train = dt_regressor.predict(X_train)
y_pred_test = dt_regressor.predict(X_test)

# Evaluate the model
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Display results
print("Decision Tree Regression Results:")
print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)
print("Training R-squared:", train_r2)
print("Testing R-squared:", test_r2)

# Feature Importance Visualization
plt.figure(figsize=(12, 6))
importances = dt_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], color="r", align="center")
plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=90)
plt.xlim([-1, len(importances)])
plt.show()

# Visualize the decision tree using plot_tree
plt.figure(figsize=(20,10)) # Adjust the figure size as needed
plot_tree(dt_regressor, filled=True, feature_names=X.columns, max_depth=3, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

