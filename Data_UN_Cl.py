# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('C:\\Users\\LENOVO\\Documents\\Intership_ML_project\\in-vehicle-coupon-recommendation.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check the data types of each column
print("\nData types of each column:")
print(data.dtypes)

# Get a summary of the dataset
print("\nSummary of the dataset:")
print(data.info())

# Task 2: Handle Missing Values
# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop columns with too many missing values (e.g., > 30% missing)
threshold = 0.3 * len(data)
data = data.dropna(thresh=threshold, axis=1)

# For categorical columns, fill missing values using the mode (most frequent value)
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)

# Specifying which columns need value transformation
# Handle inconsistent values in specified columns
value_transformations = {
    'CarryAway': {
        'never':'never',
        '1~3': 'between 1 to 3',
        '4~8': 'between 4 to 8',
        'less1': 'less than 1',
        'gt8': 'greater than 8'
    },
    'Bar': {
        'never':'never',
        '1~3': 'between 1 to 3',
        '4~8': 'between 4 to 8',
        'less1': 'less than 1',
        'gt8': 'greater than 8'
    },
    'CoffeeHouse': {
        'never':'never',
        '1~3': 'between 1 to 3',
        '4~8': 'between 4 to 8',
        'less1': 'less than 1',
        'gt8': 'greater than 8'
    },
    'RestaurantLessThan20': {
        'never':'never',
        '1~3': 'between 1 to 3',
        '4~8': 'between 4 to 8',
        'less1': 'less than 1',
        'gt8': 'greater than 8'
    },
    'Restaurant20To50': {
        'never':'never',
        '1~3': 'between 1 to 3',
        '4~8': 'between 4 to 8',
        'less1': 'less than 1',
        'gt8': 'greater than 8'
    }
}

# Handle specific transformations for specified columns
for column, transformations in value_transformations.items():
    for old_value, new_value in transformations.items():
        data[column] = data[column].replace(old_value, new_value)

# Task 3: Convert 'expiration' column to hours
def convert_expiration(expiration):
    if 'h' in expiration:
        return int(expiration.replace('h', '').strip())  # Convert '2h' to 2
    elif 'd' in expiration:
        return 24  # Convert '1d' to 24
    else:
        return np.nan  # Return NaN for unexpected formats

data['expiration'] = data['expiration'].apply(convert_expiration)

# Task 5: Handle rows without values; Drop rows where specific columns have NaN values
data.dropna(subset=['temperature', 'age', 'has_children'], inplace=True)

# Final check for missing values after cleaning
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Separate numerical and non-numerical columns
numerical_columns = ['temperature', 'expiration', 'age', 'has_children', 
                     'toCoupon_GEQ5min', 'toCoupon_GEQ15min', 
                     'toCoupon_GEQ25min', 'direction_same', 
                     'direction_opp', 'Y']

# Non-numerical columns will be the rest
non_numerical_columns = [col for col in data.columns if col not in numerical_columns]

# Display the cleaned data types
print("\nData types after cleaning:")
print(data.dtypes)

# Display the final numerical and non-numerical columns
print("\nNumerical columns:")
print(numerical_columns)

print("\nNon-numerical columns:")
print(non_numerical_columns)

# Optionally, save the cleaned data to a new CSV file
data.to_csv('C:\\Users\\LENOVO\\Documents\\Intership_ML_project\\cleaned_in_vehicle_coupon_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_in_vehicle_coupon_data.csv'.")