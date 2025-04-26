import pandas as pd
import joblib

# 1. Load the saved model
model = joblib.load("random_forest_model.joblib")  # Loading the saved random forest model

# 2. List of required feature columns
columns = [
    'Time',
    'Day_of_week',
    'Age_band_of_driver',
    'Sex_of_driver',
    'Educational_level',
    'Vehicle_driver_relation',
    'Driving_experience',
    'Lanes_or_Medians',
    'Types_of_Junction',
    'Road_surface_type',
    'Light_conditions',
    'Weather_conditions',
    'Type_of_collision',
    'Vehicle_movement',
    'Pedestrian_movement',
    'Cause_of_accident'
]

# 3. Collect user input for each feature
input_data = []
print("Please enter the following information:")

for column in columns:
    value = input(f"{column}: ")  # Asking the user for input for each feature
    input_data.append(value)  # Appending the input to the list

# 4. Convert the input data into a DataFrame
input_df = pd.DataFrame([input_data], columns=columns)  # Converting input data to a DataFrame

# 5. Make prediction using the model
predicted_severity = model.predict(input_df)  # Using the model to make a prediction

# 6. Show the result to the user
print(f"Predicted Accident Severity: {predicted_severity[0]}")  # Displaying the prediction result
