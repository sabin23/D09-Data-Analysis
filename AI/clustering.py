import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data

data = Data(1, 1)

u = data.u
e = data.e

n_clusters = 2

kmeans = KMeans(n_clusters = n_clusters)
kmeans.fit(e)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(e[:, 0], e[:, 1], c=labels, cmap='viridis', alpha=0.5, label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Signal Feature 1')
plt.ylabel('Signal Feature 2')
plt.legend()
plt.show()