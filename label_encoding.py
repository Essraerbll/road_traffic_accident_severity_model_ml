import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the Excel file
input_file = "guncel_dosya.xlsx"  # The name of your Excel file
df = pd.read_excel(input_file)

# 2. Detect all categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# 3. Apply Label Encoding to each categorical column
label_encoders = {}  # We will store the LabelEncoders for each column
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])  # Transform the column
    label_encoders[column] = le  # Store the encoder for each column

# 4. Show the labels and their corresponding numbers in each categorical column
print("Labels and their corresponding values in each categorical column:")
for column, le in label_encoders.items():
    print(f"\nLabels for column '{column}':")
    for i, label in enumerate(le.classes_):
        print(f"{i} -> {label}")

# 5. Save the updated DataFrame to a new Excel file
output_file = "label.xlsx"
df.to_excel(output_file, index=False)

print(f"The updated file has been saved as '{output_file}'.")
