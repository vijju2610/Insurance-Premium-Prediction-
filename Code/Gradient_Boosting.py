import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Selecting the features and target
features_gbm = data[['bmi', 'smoker_numeric', 'children', 'region_numeric', 'age', 
                     'gender', 'medical_history', 'family_medical_history', 
                     'exercise_frequency', 'occupation', 'coverage_level_numeric']]
target_gbm = data['charges']

# Convert categorical variables to numeric using pd.get_dummies (One-Hot Encoding)
features_gbm = pd.get_dummies(features_gbm, columns=['gender', 'medical_history', 
                                                     'family_medical_history', 
                                                     'exercise_frequency', 'occupation', 'children'])

# Splitting data into training and testing sets
X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(features_gbm, target_gbm, test_size=0.2, random_state=0)

# Creating a Gradient Boosting Regressor model
gbm_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=0)

# Fitting the model
gbm_model.fit(X_train_gbm, y_train_gbm)

# Making predictions
predictions_train_gbm = gbm_model.predict(X_train_gbm)
predictions_test_gbm = gbm_model.predict(X_test_gbm)

# Calculating RMSE to evaluate the model
rmse_train_gbm = np.sqrt(mean_squared_error(y_train_gbm, predictions_train_gbm))
rmse_test_gbm = np.sqrt(mean_squared_error(y_test_gbm, predictions_test_gbm))
print('Train Root Mean Squared Error:', rmse_train_gbm)
print('Test Root Mean Squared Error:', rmse_test_gbm)

# Calculating MAE
mae_train_gbm = mean_absolute_error(y_train_gbm, predictions_train_gbm)
mae_test_gbm = mean_absolute_error(y_test_gbm, predictions_test_gbm)
print('Train Mean Absolute Error:', mae_train_gbm)
print('Test Mean Absolute Error:', mae_test_gbm)

# Calculating MAPE
mape_train_gbm = np.mean(np.abs((y_train_gbm - predictions_train_gbm) / y_train_gbm)) * 100
mape_test_gbm = np.mean(np.abs((y_test_gbm - predictions_test_gbm) / y_test_gbm)) * 100
print('Train Mean Absolute Percentage Error:', mape_train_gbm)
print('Test Mean Absolute Percentage Error:', mape_test_gbm)

# Calculating RSE
sse_train_gbm = np.sum((predictions_train_gbm - y_train_gbm) ** 2)
sst_train_gbm = np.sum((y_train_gbm - np.mean(y_train_gbm)) ** 2)
rse_train_gbm = sse_train_gbm / sst_train_gbm
sse_test_gbm = np.sum((predictions_test_gbm - y_test_gbm) ** 2)
sst_test_gbm = np.sum((y_test_gbm - np.mean(y_test_gbm)) ** 2)
rse_test_gbm = sse_test_gbm / sst_test_gbm
print('Train Relative Squared Error:', rse_train_gbm)
print('Test Relative Squared Error:', rse_test_gbm)

# Feature importances
feature_importances_gbm = gbm_model.feature_importances_
print('Feature Importances:', feature_importances_gbm)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Selecting features and target
features = data[['bmi', 'smoker_numeric', 'children', 'region_numeric', 'age', 'gender',
                 'medical_history', 'family_medical_history', 'exercise_frequency', 
                 'occupation', 'coverage_level_numeric']]
target = data['charges']

# Convert categorical variables using pd.get_dummies (One-Hot Encoding)
features = pd.get_dummies(features, columns=['gender', 'medical_history', 'family_medical_history',
                                             'exercise_frequency', 'occupation', 'children'])

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA (retain 90% of the variance)
pca = PCA(n_components=0.90)
features_pca = pca.fit_transform(features_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=0)

# XGBoost regression model
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, 
                         max_depth=10, alpha=15, n_estimators=500)
model.fit(X_train, y_train)

# Make predictions
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Evaluate the model
rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))
rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))
mae_train = mean_absolute_error(y_train, predictions_train)
mae_test = mean_absolute_error(y_test, predictions_test)
r2_train = r2_score(y_train, predictions_train)
r2_test = r2_score(y_test, predictions_test)

print('Train RMSE:', rmse_train)
print('Test RMSE:', rmse_test)
print('Train MAE:', mae_train)
print('Test MAE:', mae_test)
print('Train R²:', r2_train)
print('Test R²:', r2_test)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming 'data' is your DataFrame
categorical_features = ['gender', 'medical_history', 'family_medical_history', 'occupation',
'coverage_level_numeric', 'children']

features = data.drop('charges', axis=1) # all columns except target

# Using ColumnTransformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), ['bmi', 'age']), #Assuming 'age' and 'bmi' are numeric
('cat', OneHotEncoder(), categorical_features)])

# Now fit_transform your data
features_preprocessed = preprocessor.fit_transform(features)
from sklearn.model_selection import GridSearchCV

# Define the model
model = xgb.XGBRegressor(tree_method='hist')

# Set up the parameter grid
param_grid = {'max_depth': [10, 15, 20],'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100,
500,1000],'alpha':[10,15,20],}

# Configure GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3,
scoring='neg_mean_squared_error',n_jobs=-1,verbose=1)
grid_search.fit(features_preprocessed,data['charges'])

# Print best parameters and lowest RMSE
print("Best parameters found: ", grid_search.best_params_)
print("Best RMSE: ", np.sqrt(-grid_search.best_score_))
