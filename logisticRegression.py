import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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

# 4. Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
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
plt.bar(["Logistic Regression"], scores, color='purple')
plt.ylabel("Accuracy")
plt.title("Logistic Regression Model Accuracy")
plt.show()
