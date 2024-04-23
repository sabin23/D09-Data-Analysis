import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
# from skfda import datasets
# from skfda.exploratory.visualization.clustering import (
#     ClusterMembershipLinesPlot,
#     ClusterMembershipPlot,
#     ClusterPlot,
# )
# from skfda.ml.clustering import FuzzyCMeans, KMeans

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data

data = Data(2, 5)

e = data.e
x = data.x
u = data.u
t = data.t

rms = np.sqrt(np.mean(e**2))

e = np.mean(e, axis=1)[:, np.newaxis]
x = np.mean(x, axis=1)[:, np.newaxis]
u = np.mean(u, axis=1)[:, np.newaxis]
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

# ------------------- Elbow method -------------------
# distortions = []
# K = range(1, 11)
# for k in K:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(np.concatenate((e, xdot), axis=1))
#     distortions.append(kmeans.inertia_)

# # Find the elbow point
# rate_of_change = [abs(distortions[i] - distortions[i-1]) for i in range(1, len(distortions))]

# elbow_index = rate_of_change.index(max(rate_of_change)) + 1

# optimal_k = K[elbow_index]

# print("Optimal number of clusters:", optimal_k)
# print("Rate of change:", rate_of_change)

# # Plot the elbow curve
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Distortion')
# plt.title('Elbow Method')
# plt.show()

# 2D combined data
combined_data = np.concatenate((e, xdot), axis=1)
# 3D combined data
# combined_data = np.concatenate((e, xdot, u), axis=1)

n_clusters = 2

# ------------------- Using AI Course -------------------

def train(train_dict, data_dict, n_clusters):
    X_train = []
    for i in range(len(train_dict)):
        X_train.extend(data_dict[f'a{i + 1}']['strains'])

    X_train = np.array(X_train).squeeze().reshape(-1, 1)

    X_train = np.sort(X_train.reshape(-1))

    cluster_model = Kmeans(n_clusters=n_clusters, n_init=10).fit(X_train.reshape(-1, 1))

    _, indx = np.unique(cluster_model.labels_, return_index=True)
    lookup_labels = [cluster_model.labels_[i] for i in sorted(indx)]

    return cluster_model, lookup_labels

def predict(cluster_model, lookup_labels, data_dict, cluster_method, train):
    if train:
        clusters_dict = {f'a{i + 1}': [] for i in range(len(6))}
    else:
        clusters_dict = {f'a{i + 1}': [] for i in range(len(2))}

    for i in range(len(data_dict)):
        if train:
            pred = cluster_model.predict(
                np.array(data_dict[f'a{i + 1}']['strains']).squeeze().reshape(-1, 1))
        else:
            pred = cluster_model.predict(
                np.array(data_dict[f'te_a{i + 1}']['strains']).squeeze().reshape(-1, 1))
        labels = []
        for j in range(len(pred)):
            labels.append(np.where(lookup_labels == pred[j])[0])
        if train:
            clusters_dict[f'a{i + 1}'].extend(sorted(labels))
        else:
            clusters_dict[f'te_a{i + 1}'].extend(sorted(labels))
    
    return clusters_dict

train_dict = e
test_dict = e

data_dict = {}
data_dict.update(train_dict)
data_dict.update(test_dict)

cluster_model, lookup_labels = train(data.data_dict, data.data_dict, n_clusters)

train_labels_dict = predict(cluster_model, lookup_labels, train_dict, train=True)
test_labels_dict = predict(cluster_model, lookup_labels, test_dict, train=False)

# ------------------- K-means -------------------

kmeans = KMeans(n_clusters = n_clusters)
kmeans.fit(combined_data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

centroids_diff = np.linalg.norm(centroids[0] - centroids[1])

print("Scalar difference between centroids:", centroids_diff)
print("RMS e:", rms)

# ------------------- 3D K-means -------------------

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# scatter1 = ax.scatter(combined_data[labels == 0, 0], combined_data[labels == 0, 1], combined_data[labels == 0, 2], 
#                      c='yellow', alpha=0.5, label='Cluster 1')
# scatter2 = ax.scatter(combined_data[labels == 1, 0], combined_data[labels == 1, 1], combined_data[labels == 1, 2], 
#                      c='purple', alpha=0.5, label='Cluster 2')

# centroid1 = ax.scatter(centroids[0, 0], centroids[0, 1], centroids[0, 2], marker='o', s=200, c='red', label='Centroid 1')
# centroid2 = ax.scatter(centroids[1, 0], centroids[1, 1], centroids[1, 2], marker='o', s=200, c='blue', label='Centroid 2')

# # Show legend
# plt.legend(handles=[scatter1, scatter2, centroid1, centroid2])

# ax.set_xlabel('e')
# ax.set_ylabel('xdot')
# ax.set_zlabel('u')
# ax.set_title('K-means Clustering')
# plt.legend()
# plt.show()

# ------------------- 2D K-means -------------------

scatter1 = plt.scatter(combined_data[labels == 0, 0], combined_data[labels == 0, 1],
                     c='green', alpha=0.5, label='Cluster 1')
scatter2 = plt.scatter(combined_data[labels == 1, 0], combined_data[labels == 1, 1],
                     c='purple', alpha=0.5, label='Cluster 2')
scatter3 = plt.scatter(combined_data[labels == 2, 0], combined_data[labels == 2, 1],
                     c='blue', alpha=0.5, label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('e')
plt.ylabel('$\dot{x}$')
plt.legend()
plt.show()


# ------------------- Hierarchical Clustering -------------------

# ------------------- Agglomerative Clustering ------------------   

# hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
# hierarchical.fit(combined_data)
# h_labels = hierarchical.labels_

# # Plot the hierarchical clustering results
# scatter1 = plt.scatter(combined_data[h_labels == 0, 0], combined_data[h_labels == 0, 1],
#                      c='pink', alpha=0.5, label='Cluster 1')
# scatter2 = plt.scatter(combined_data[h_labels == 1, 0], combined_data[h_labels == 1, 1],
#                      c='green', alpha=0.5, label='Cluster 2')
# scatter3 = plt.scatter(combined_data[h_labels == 2, 0], combined_data[h_labels == 2, 1],
#                      c='blue', alpha=0.5, label='Cluster 3')
# plt.title('Hierarchical Clustering')
# plt.xlabel('e')
# plt.ylabel('$\dot{x}$')
# plt.legend()
# plt.show()

# ------------------- Divisive Clustering -------------------

# combined_data, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# divisive = AgglomerativeClustering(n_clusters=n_clusters)
# divisive.fit(combined_data)
# d_labels = divisive.labels_

# # Plot the divisive clustering results
# scatter1 = plt.scatter(combined_data[d_labels == 0, 0], combined_data[d_labels == 0, 1],
#                      c='green', alpha=0.5, label='Cluster 1')
# scatter2 = plt.scatter(combined_data[d_labels == 1, 0], combined_data[d_labels == 1, 1],
#                      c='purple', alpha=0.5, label='Cluster 2')
# scatter3 = plt.scatter(combined_data[d_labels == 2, 0], combined_data[d_labels == 2, 1],
#                      c='blue', alpha=0.5, label='Cluster 3')
# plt.title('Divisive Clustering')
# plt.xlabel('e')
# plt.ylabel('$\dot{x}$')
# plt.legend()
# plt.show()