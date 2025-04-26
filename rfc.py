import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Import joblib for saving the model

# 1. Load the dataset with SMOTE applied
file_name = "smote_label.xlsx"  # The file with SMOTE applied
df = pd.read_excel(file_name)

# 2. Separate features and target variable
target_column = 'Accident_severity'  # The name of the target variable
X = df.drop(columns=[target_column])  # Features
y = df[target_column]                # Target variable

# 3. Split the data into Training and Test Sets (70% Training, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 4. Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions on the Test Set
y_pred = model.predict(X_test)

# 6. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# 7. Detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Visualizing the model's performance (Accuracy)
scores = [accuracy]

# Visualizing the accuracy using a bar plot
plt.bar(["Random Forest"], scores, color='green')
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.show()

# 9. Save the model
model_file_name = "random_forest_model.joblib"  # Name of the model file
joblib.dump(model, model_file_name)
print(f"Model successfully saved as {model_file_name}.")

# 10. Feature Importance Plot
importances = model.feature_importances_
# Sorting the features
indices = importances.argsort()[::-1]

# Get the feature names
feature_names = X.columns

# Visualizing the feature importance
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(len(importances)), importances[indices], align="center")
plt.yticks(range(len(importances)), feature_names[indices])
plt.xlabel("Feature Importance")
plt.show()
