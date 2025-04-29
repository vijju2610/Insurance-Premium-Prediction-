from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Initialize Logistic Regression model
lr_classifier = LogisticRegression(random_state=42, max_iter=1000)

# Train the Logistic Regression model
lr_classifier.fit(X_train, y_train)

# Make predictions with Logistic Regression
y_pred_train_lr = lr_classifier.predict(X_train)
y_pred_test_lr = lr_classifier.predict(X_test)

# Evaluate the Logistic Regression model
train_accuracy_lr = accuracy_score(y_train, y_pred_train_lr)
test_accuracy_lr = accuracy_score(y_test, y_pred_test_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_test_lr)
class_report_lr = classification_report(y_test, y_pred_test_lr)

# Display Logistic Regression results
print("Logistic Regression Classification Results:")
print("Training Accuracy:", train_accuracy_lr)
print("Testing Accuracy:", test_accuracy_lr)
print("Confusion Matrix:\n", conf_matrix_lr)
print("Classification Report:\n", class_report_lr)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the Random Forest model
rf_classifier.fit(X_train, y_train)

# Make predictions with Random Forest
y_pred_train_rf = rf_classifier.predict(X_train)
y_pred_test_rf = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
train_accuracy_rf = accuracy_score(y_train, y_pred_train_rf)
test_accuracy_rf = accuracy_score(y_test, y_pred_test_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_test_rf)
class_report_rf = classification_report(y_test, y_pred_test_rf)

# Display Random Forest results
print("Random Forest Classification Results:")
print("Training Accuracy:", train_accuracy_rf)
print("Testing Accuracy:", test_accuracy_rf)
print("Confusion Matrix:\n", conf_matrix_rf)
print("Classification Report:\n", class_report_rf)
