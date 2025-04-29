from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define categorical and numerical features
categorical_features = ['smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level']
numeric_features = ['age', 'bmi', 'children']

# Prepare features and target variable
X = data[categorical_features + numeric_features]
y = data['charges']

# Prepare features and target variable
X = data[categorical_features + numeric_features]
y = data['charges']

# Encoding categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")

# Feature importances
feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
importances = model.feature_importances_
feature_importance_dict = dict(zip(feature_names, importances))

# Print feature importances
print("Feature Importances:")
for name, importance in sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True):
    print(f"{name}: {importance:.4f}")

import matplotlib.pyplot as plt

# Sorting feature importances
sorted_importances = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
sorted_features = [x[0] for x in sorted_importances]
sorted_scores = [x[1] for x in sorted_importances]

# Creating the bar plot
plt.figure(figsize=(10, 8))
plt.barh(sorted_features, sorted_scores, color='skyblue')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance in Random Forest Regressor')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()

from sklearn.metrics import r2_score, mean_absolute_error

#calculated RMSE:
print(f"Root Mean Squared Error: {rmse}")

# Calculating R-squared and MAE:
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")

from sklearn.metrics import r2_score, mean_absolute_error

# Assuming 'y_test' and 'y_pred' are the actual and predicted values for the test set
r2_test = r2_score(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)

print(f"Test R-squared: {r2_test}")
print(f"Test Mean Absolute Error: {mae_test}")

# Predict on the training set
y_train_pred = model.predict(X_train)

# Calculate metrics for the training set to check for overfitting
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)

print(f"Train R-squared: {r2_train}")
print(f"Train Mean Absolute Error: {mae_train}")

# Compare the performance metrics of the training set and the test set
if r2_train > r2_test:
    if (r2_train - r2_test) > 0.1:  # Arbitrary threshold for significant difference
        print("The model may be overfitting to the training data.")
    else:
        print("The model is not significantly overfitting.")
else:
    print("The model is generalizing well to unseen data.")

