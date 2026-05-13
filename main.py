import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("customers.csv")

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Find clusters
kmeans = KMeans(n_clusters=5, random_state=42)

# Train model
kmeans.fit(X)

# Predict clusters
data['Cluster'] = kmeans.predict(X)

# Plot graph
plt.figure(figsize=(8,6))

plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=data['Cluster'],
    cmap='rainbow'
)

# Cluster centers
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    c='black',
    label='Centroids'
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.legend()
plt.show()