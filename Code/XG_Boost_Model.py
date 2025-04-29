import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Selecting the features and target
features = data[['bmi', 'smoker_numeric', 'children',
'region_numeric', 'age', 'gender', 'medical_history', 'family_medical_history',
'exercise_frequency', 'occupation', 'coverage_level_numeric']]
target = data['charges']

# Convert categorical variables using pd.get_dummies (One-Hot Encoding)
features = pd.get_dummies(features, columns=['gender', 'medical_history', 'family_medical_history',
                                             'exercise_frequency', 'occupation', 'children'])

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Creating an XGBoost regression model
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=20, alpha=15, n_estimators=1000)

# Fitting the model
model.fit(X_train, y_train)

# Making predictions
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Calculating RMSE to evaluate the model
rmse_test = mean_squared_error(y_test, predictions_test, squared=False)
rmse_train = mean_squared_error(y_train, predictions_train, squared=False)

# Getting feature importances
feature_importances = model.feature_importances_

print('Train Root Mean Squared Error:', rmse_train)
print('Test Root Mean Squared Error:', rmse_test)
print('Feature Importances:', feature_importances)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Calculating MAE
mae_test = mean_absolute_error(y_test, predictions_test)
mae_train=mean_absolute_error(y_train,predictions_train)
print('Train Mean Absolute Error:', mae_train)
print('Test Mean Absolute Error:', mae_test)

# Calculating MAPE
mape_test = np.mean(np.abs((y_test - predictions_test) / y_test)) * 100
mape_train= np.mean(np.abs((y_train - predictions_train) / y_train)) * 100
print('Train Mean Absolute Percentage Error:', mape_train)
print('Test Mean Absolute Percentage Error:', mape_test)

# Calculating RSE
sse_test = np.sum((predictions_test - y_test) ** 2)
sst_test = np.sum((y_test - np.mean(y_test)) ** 2)
rse_test = sse_test / sst_test
sse_train = np.sum((predictions_train - y_train) ** 2)
sst_train = np.sum((y_train - np.mean(y_train)) ** 2)
rse_train = sse_train / sst_train

print('Train Relative Squared Error:', rse_train)
print('Test Relative Squared Error:', rse_test)

# Assuming y_test holds the actual values and predictions holds the predicted values from your model
r_squared_test = r2_score(y_test, predictions_test)
r_squared_train = r2_score(y_train, predictions_train)
print("Train R-squared Value:", r_squared_train)
print("Test R-squared Value:", r_squared_test)

import matplotlib.pyplot as plt
# Assuming 'features' is your DataFrame and you have feature names there
feature_names = features.columns.tolist() # This dynamically captures all feature names

# Let's assume these are the importances you retrieved from your model
feature_importances = [9.5288378e-06 ,9.7213435e-01, 1.3350404e-05 ,6.6113330e-06, 2.1387734e-02,
 3.0389019e-05 ,3.6896909e-05 ,6.5625936e-05, 2.3437806e-03, 5.1778852e-04,
 6.9721806e-05 ,2.3867460e-03 ,1.8811238e-04, 2.7109083e-04, 5.9400201e-05,
 1.3783697e-05 ,1.9660411e-05, 4.6544632e-05, 1.5173599e-05, 1.5327665e-04,
 1.4013713e-04 ,2.5480698e-05, 6.3124448e-06 ,3.4317482e-06, 7.3175661e-06,
 1.8354272e-05, 2.9287421e-05]
# Plotting
plt.figure(figsize=(12, 8))
plt.barh(feature_names, feature_importances, color='blue')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis() # Invert y-axis to have the most important feature on top
plt.show()

residuals = y_test - predictions_test
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Actual Charges')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4) # Line representing perfect predictions
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs. Predicted Charges')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=20, alpha=0.5, label='Actual Charges', color='darkblue')
plt.hist(predictions_test, bins=20, alpha=0.5, label='Predicted Charges', color='darkred')
plt.xlabel('Charge Amount')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Actual and Predicted Charges')
plt.show()

