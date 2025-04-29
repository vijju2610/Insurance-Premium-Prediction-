import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define categorical and numerical features
categorical_features = ['smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level']
numeric_features = ['age', 'bmi', 'children']

# Prepare features and target variable
X = data[categorical_features + numeric_features]
y = data['charges']


# Encoding categorical variables and scaling numerical variables
column_transformer = ColumnTransformer(
    [('num', StandardScaler(), numeric_features),
     ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
pipeline = Pipeline([
    ('preprocessor', column_transformer),
    ('regressor', LinearRegression())
])

# Fit the model
pipeline.fit(X_train, y_train)

Pipeline(steps=[('preprocessor',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('num', StandardScaler(),
                                                  ['age', 'bmi', 'children']),
                                                 ('cat',
                                                  OneHotEncoder(handle_unknown='ignore'),
                                                  ['smoker', 'region',
                                                   'medical_history',
                                                   'family_medical_history',
                                                   'exercise_frequency',
                                                   'occupation',
                                                   'coverage_level'])])),
                ('regressor', LinearRegression())])
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# Evaluate the model
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Output model performance metrics
print(f"Training R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Cross-validation to assess model robustness
cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f"Average cross-validation R²: {np.mean(cv_r2):.2f}")

# Residual plots for diagnostics
plt.figure(figsize=(10, 5))
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual Charges')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Plot actual vs. predicted charges
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.show()
