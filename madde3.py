import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Iris veri setini yükleyelim
iris_data = pd.read_csv("Iris.csv")

# Veri setini inceleyelim
print(iris_data.head())

# Veriyi bağımsız değişkenler (X) olarak alalım
X = iris_data.drop("Species", axis=1)

# K-means için optimal küme sayısını belirleme (Elbow Method)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Elbow Method grafiği
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Optimal k değerini seçelim (Elbow noktası)
optimal_k = 3  # Bu örnekte, grafikten anlaşılacağı üzere, optimal küme sayısı 3 gibi görünüyor.

# K-means modelini oluşturalım
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)
iris_data["KMeans_Cluster"] = kmeans_model.fit_predict(X)

# Hiyerarşik Kümeleme (Agglomerative Clustering) için dendrogram
plt.figure(figsize=(12, 6))
sns.clustermap(X, method='ward', cmap='viridis', figsize=(12, 8))
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()

# Hiyerarşik Kümeleme modelini oluşturalım
agglomerative_model = AgglomerativeClustering(n_clusters=optimal_k)
iris_data["Hierarchical_Cluster"] = agglomerative_model.fit_predict(X)

# Kümeleme sonuçlarını inceleyelim
print("K-means Clusters:")
print(iris_data["KMeans_Cluster"].value_counts())

print("\nHierarchical Clusters:")
print(iris_data["Hierarchical_Cluster"].value_counts())

# Silhouette skoru ile kümeleme performansını değerlendirelim
silhouette_kmeans = silhouette_score(X, iris_data["KMeans_Cluster"])
silhouette_hierarchical = silhouette_score(X, iris_data["Hierarchical_Cluster"])

print("\nSilhouette Score for K-means:", silhouette_kmeans)
print("Silhouette Score for Hierarchical Clustering:", silhouette_hierarchical)
