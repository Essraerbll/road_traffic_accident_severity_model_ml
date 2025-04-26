import pandas as pd

# 1. Load the Excel file
df = pd.read_excel('guncel_dosya.xlsx', engine='openpyxl')

# 2. Replace 'unknown' and 'Unknown' values with NaN in categorical columns
for column in df.columns:
    # Only process categorical (string) columns
    if df[column].dtype == 'object':
        # Replace 'unknown' and 'Unknown' with NaN
        df[column] = df[column].replace(['unknown', 'Unknown'], pd.NA)
        
        # Find the mode (most frequent value), ignoring NaN values
        mode_value = df[column].mode()[0]
        
        # Fill NaN values with the mode value
        df[column] = df[column].fillna(mode_value)

# 3. Save the updated DataFrame back to the Excel file
df.to_excel('guncel_dosya.xlsx', index=False, engine='openpyxl')

# 4. Notify the user that the file has been updated
print("The file has been updated and saved as 'guncel_dosya.xlsx'.")
