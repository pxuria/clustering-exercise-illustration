import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for environments without GUI support

# Step 1: Load CSV Data
file_path = './dataset/winequality-red.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path, sep=';')  # Ensure proper separator is used

# Step 2: Data Preprocessing
X = data.values  # Extract the feature values (assuming no target column)

# Normalize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Define a function for evaluation
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

# Step 4: Perform CLINK (Complete-Linkage Clustering)
clink_linkage = linkage(X_scaled, method='complete')  # Complete-linkage
clink_labels = fcluster(clink_linkage, t=3, criterion='maxclust')  # Adjust 't' for desired number of clusters
evaluate_clustering(X_scaled, clink_labels, "CLINK (Complete-Linkage)")

# Step 5: Perform SLINK (Single-Linkage Clustering)
slink_linkage = linkage(X_scaled, method='single')  # Single-linkage
slink_labels = fcluster(slink_linkage, t=3, criterion='maxclust')  # Adjust 't' for desired number of clusters
evaluate_clustering(X_scaled, slink_labels, "SLINK (Single-Linkage)")

# Step 6: Visualize Dendrograms
# Function to save dendrograms as images
def save_dendrogram(linkage_matrix, method_name, file_name):
    plt.figure(figsize=(10, 5))
    plt.title(f"Dendrogram for {method_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    dendrogram(linkage_matrix)
    plt.savefig(file_name)  # Save the plot as an image
    print(f"{method_name} dendrogram saved as {file_name}")
    plt.close()

# Save dendrograms
save_dendrogram(clink_linkage, "CLINK (Complete-Linkage)", "clink_dendrogram.png")
save_dendrogram(slink_linkage, "SLINK (Single-Linkage)", "slink_dendrogram.png")
