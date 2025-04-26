import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. Load the Excel file
file_name = "label.xlsx"
df = pd.read_excel(file_name)

# 2. Separate features and target variable
target_column = 'Accident_severity'  # Name of the target variable
X = df.drop(columns=[target_column])  # Features
y = df[target_column]                # Target variable

# 3. Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. Check class distribution before and after SMOTE
print("Original class distribution:", Counter(y))
print("Class distribution after SMOTE:", Counter(y_resampled))

# 5. Create plots for before and after SMOTE
for column in X.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{column} Column Before and After SMOTE', fontsize=16)
    
    # Plot before SMOTE
    axes[0].hist(X[column], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Before SMOTE')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    
    # Plot after SMOTE
    axes[1].hist(X_resampled[column], bins=30, color='salmon', edgecolor='black')
    axes[1].set_title('After SMOTE')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')
    
    # Save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{column}_smote_plot.png')  # Save a separate file for each column
    plt.show()

# 6. Combine the new dataset and save
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                          pd.DataFrame(y_resampled, columns=[target_column])], axis=1)

output_file = "smote_label.xlsx"
df_resampled.to_excel(output_file, index=False)
print(f"SMOTE-applied file saved as '{output_file}'.")
