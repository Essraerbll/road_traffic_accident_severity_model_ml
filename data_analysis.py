import pandas as pd

# 1. Reading the Excel File
file_path = r"example.path"  # Original file path
df = pd.read_excel(file_path, sheet_name='Sheet1')  # Read the first sheet

# 2. Analyze Missing Data Rates and Unique Values
print("Column Analysis: Unique Values, Missing Data Rates\n")
for col in df.columns:
    total_values = len(df)  # Total number of rows
    unique_values = df[col].dropna().unique()  # Find unique values excluding missing data
    num_unique = len(unique_values)  # Count of unique values
    missing_count = df[col].isnull().sum()  # Count of missing values
    missing_percent = (missing_count / total_values) * 100  # Percentage of missing values

    print(f"Column Name: {col}")
    print(f" - Number of Unique Values: {num_unique}")
    print(f" - Unique Values: {list(unique_values)}")
    print(f" - Number of Missing Values: {missing_count}")
    print(f" - Missing Data Percentage: %{missing_percent:.2f}")
    print("-" * 50)


