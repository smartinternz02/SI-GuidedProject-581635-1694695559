import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\prana\Downloads\Mall_Customers.csv")  # Replace with your dataset file path

# Data Cleaning (Handling missing values)
data.dropna(inplace=True)  # Remove rows with missing values

# Feature Selection/Engineering (Select relevant columns)
selected_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
data = data[selected_features]

# Data Scaling/Normalization (Standardization)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph to find the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Based on the Elbow method, choose the optimal K (number of clusters)
optimal_k = 5  # Adjust based on the Elbow plot

# Perform K-Means clustering with the optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = clusters

# Print the entire dataset with cluster labels
print(data)
