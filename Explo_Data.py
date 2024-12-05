# Import necessary libraries for data visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
data = pd.read_csv('cleaned_in_vehicle_coupon_data.csv')

# Set the visual style of seaborn
sns.set(style='whitegrid')

# Task 1: Visualize Relationships
# Visualize relationships between target variable (Y) and categorical features

# This will show how coupon acceptance (Y) varies with different weather conditions
plt.figure(figsize=(10, 6))
sns.countplot(x='weather', hue='Y', data=data, palette='Set1')
plt.title('Coupon Acceptance by Weather')
plt.xlabel('Weather Conditions')
plt.ylabel('Count')
plt.legend(title='Coupon Accepted', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize relationships between target variable (Y) and passenger type
plt.figure(figsize=(10, 6))
sns.countplot(x='passenger', hue='Y', data=data, palette='Set2')
plt.title('Coupon Acceptance by Passenger Type')
plt.xlabel('Passenger Type')
plt.ylabel('Count')
plt.legend(title='Coupon Accepted', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize relationships between target variable (Y) and time of day
plt.figure(figsize=(10, 6))
sns.countplot(x='time', hue='Y', data=data, palette='Set3')
plt.title('Coupon Acceptance by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Count')
plt.legend(title='Coupon Accepted', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize relationships between target variable (Y) and age group
plt.figure(figsize=(10, 6))
sns.countplot(x='age', hue='Y', data=data, palette='pastel')
plt.title('Coupon Acceptance by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Coupon Accepted', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 2: Analyze Trends
# Analyze trends in coupon acceptance based on categorical features
# This will show the average coupon acceptance rate for each time of day
plt.figure(figsize=(10, 6))
sns.barplot(x='time', y='Y', data=data, estimator=lambda x: sum(x == 'Yes') / len(x), hue='time', palette='viridis')
plt.title('Average Coupon Acceptance by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Average Coupon Acceptance Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 3: Correlation Analysis
# Perform correlation analysis between numerical features and the target variable (Y)

# First, filter out non-numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Task 4: Visualize Distributions
# Visualize distributions of key numerical features

# Visualizing the distribution of temperature
plt.figure(figsize=(10, 6))
sns.histplot(data['temperature'], bins=30, kde=True, color='blue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Visualizing the distribution of toCoupon_GEQ5min
plt.figure(figsize=(10, 6))
sns.histplot(data['toCoupon_GEQ5min'], bins=30, kde=True, color='orange')
plt.title('Distribution of Time to Coupon (>= 5 minutes)')
plt.xlabel('Time to Coupon (>= 5 minutes)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Visualizing the distribution of toCoupon_GEQ15min
plt.figure(figsize=(10, 6))
sns.histplot(data['toCoupon_GEQ15min'], bins=30, kde=True, color='green')
plt.title('Distribution of Time to Coupon (>= 15 minutes)')
plt.xlabel('Time to Coupon (>= 15 minutes)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Visualizing the distribution of toCoupon_GEQ25min
plt.figure(figsize=(10, 6))
sns.histplot(data['toCoupon_GEQ25min'], bins=30, kde=True, color='red')
plt.title('Distribution of Time to Coupon (>= 25 minutes)')
plt.xlabel('Time to Coupon (>= 25 minutes)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
