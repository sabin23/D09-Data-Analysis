import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data

data = Data(2, 1)

e = data.e
x = data.x
t = data.t

rms = np.sqrt(np.mean(e**2))

# combined_data = np.mean(np.concatenate((x, xdot), axis=0), axis=1)[:, np.newaxis]

e = np.mean(e, axis=1)[:, np.newaxis]
x = np.mean(x, axis=1)[:, np.newaxis]
t = t.reshape(-1, 1)

def dfdt (f, t):
    dfdt = np.zeros(f.shape)
    for i in range(0, len(f)-1):
        if i > 0 and i < len(f)-1:
            dfdt[i] = (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
    dfdt[len(f)-1] = dfdt[len(f)-2]
    dfdt[0] = dfdt[1]
    return dfdt

xdot = dfdt(x, t)

xdot = np.mean(xdot, axis=1)[:, np.newaxis]
combined_data = np.concatenate((e, xdot), axis=1)

n_clusters = 2

kmeans = KMeans(n_clusters = n_clusters)
kmeans.fit(combined_data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

centroids_diff = np.linalg.norm(centroids[0] - centroids[1])

print("Scalar difference between centroids:", centroids_diff)
print("RMS e:", rms)

plt.scatter(combined_data[:, 0], combined_data[:, 1], c=labels, cmap='viridis', alpha=0.5, label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()