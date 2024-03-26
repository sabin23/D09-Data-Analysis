import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data

data1 = Data(2, 1)
data2 = Data(2, 2)

e1 = data1.e
e2 = data2.e
t = data1.t
x = data1.x



rms_1 = np.sqrt(np.mean(e1**2))
rms_2 = np.sqrt(np.mean(e2**2))

# combined_data = np.mean(np.concatenate((e1, e2), axis=0), axis=1)[:, np.newaxis]

e2 = np.mean(e2, axis=1)[:, np.newaxis]

# mf_indicator = np.concatenate((np.zeros((e1.shape[0], e1.shape[1])), np.ones((e2.shape[0], e2.shape[1]))), axis=0)
# correlation = np.corrcoef(combined_data, mf_indicator)[0, 1]

# features = np.concatenate((np.abs(combined_data), correlation[:, np.newaxis]), axis=1)

# features = np.abs(combined_data)

features = np.abs(e2)

n_clusters = 2

kmeans = KMeans(n_clusters = n_clusters)
kmeans.fit(features)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

centroids_diff = np.linalg.norm(centroids[0] - centroids[1])

print("Scalar difference between centroids:", centroids_diff)
print("RMS e 1:", rms_1)
print("RMS e 2:", rms_2)

plt.scatter(t, e2, c=labels, cmap='viridis', alpha=0.5, label='Data Points')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('t')
plt.ylabel('e2')
plt.legend()
plt.show()