import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(r"C:\Users\prana\Downloads\Mall_Customers.csv")  # Replace with your dataset file path

# Data Cleaning (Handling missing values)
data.dropna(inplace=True)  # Remove rows with missing values
print("Data after cleaning:")
print(data.head())  # Print the first few rows of the cleaned data

# Feature Selection/Engineering (Select relevant columns)
selected_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
data = data[selected_features]
print("\nData after feature selection/engineering:")
print(data.head())  # Print the first few rows of the selected features

# Data Scaling/Normalization (Standardization)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Rest of your code for clustering...






