import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Step 1: Load CSV Data
file_path = './dataset/winequality-red.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path, sep=";")

# Step 2: Data Preprocessing
# Assuming that the dataset has numerical values and no categorical columns that need encoding
X = data.values  # Extract the feature values (assuming no target column)

# Normalize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # You can adjust n_clusters based on your dataset
kmeans_labels = kmeans.fit_predict(X_scaled)

# Step 4: Evaluate Clustering
def evaluate_clustering(data, labels, method_name):
    if len(set(labels)) > 1:  # Ensure more than 1 cluster is detected
        sil_score = silhouette_score(data, labels)
        calinski_score = calinski_harabasz_score(data, labels)
        davies_score = davies_bouldin_score(data, labels)
        print(f"{method_name}:")
        print(f"  Silhouette Score: {sil_score:.3f}")
        print(f"  Calinski-Harabasz Index: {calinski_score:.3f}")
        print(f"  Davies-Bouldin Index: {davies_score:.3f}")
    else:
        print(f"{method_name}: Only one cluster detected.")

# Evaluate KMeans
evaluate_clustering(X_scaled, kmeans_labels, "KMeans")

# Step 5: Visualize KMeans Clustering
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Save the plot to a file (optional)
plt.savefig("kmeans_clustering.png")
plt.close()
