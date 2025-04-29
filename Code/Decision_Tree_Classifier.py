from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Encode all categorical variables to numeric using LabelEncoder
label_encoders = {}
categorical_columns = ['gender', 'smoker', 'region', 'medical_history', 'family_medical_history',
'exercise_frequency', 'occupation', 'coverage_level']
for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le # Store each encoder if needed later for inverse transform or validation

print(data.head())

# Define features and target for classification
X = data.drop('coverage_level', axis=1) # Drop the target column to isolate features
y = data['coverage_level'] # Use the encoded 'coverage_level' as the target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_train = dt_classifier.predict(X_train)
y_pred_test = dt_classifier.predict(X_test)

# Evaluate the classifier
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)
class_report = classification_report(y_test, y_pred_test)

# Display classification results
print("Decision Tree Classification Results:")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Visualize the decision tree using plot_tree
plt.figure(figsize=(20,10)) # Adjust the figure size as needed
plot_tree(dt_classifier, filled=True, feature_names=X.columns, max_depth=3, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

