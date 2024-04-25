import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data

# nm -> no motion; m -> motion; (pilot, test)
data_nm = Data(3, 5)
data_m = Data(3, 6)

e_nm = data_nm.e
e_m = data_m.e

x_nm = data_nm.x
x_m = data_m.x

u_nm = data_nm.u
u_m = data_m.u

t = data_nm.t

def rms(f):
    return np.sqrt((np.mean(f, axis=0))**2).reshape(-1, 1)

rms_e_nm = rms(e_nm)
rms_e_m = rms(e_m)

rms_u_nm = rms(u_nm)
rms_u_m =  rms(u_m)

t = t.reshape(-1, 1)

def dfdt (f, t):
    dfdt = np.zeros(f.shape)
    for i in range(0, len(f)-1):
        if i > 0 and i < len(f)-1:
            dfdt[i] = (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
    dfdt[len(f)-1] = dfdt[len(f)-2]
    dfdt[0] = dfdt[1]
    return dfdt

edot_nm = dfdt(e_nm, t)
edot_m = dfdt(e_m, t)

udot_nm = dfdt(u_nm, t)
udot_m = dfdt(u_m, t)

def mean(f):
    return np.mean(f, axis=1).reshape(-1, 1)

e_nm = mean(e_nm)
e_m = mean(e_m)

u_nm = mean(u_nm)
u_m = mean(u_m)


rms_edot_nm = rms(edot_nm)
rms_edot_m = rms(edot_m)

rms_udot_nm = rms(udot_nm)
rms_udot_m = rms(udot_m)


edot_nm = mean(edot_nm)
edot_m = mean(edot_m)

udot_nm = mean(udot_nm)
udot_m = mean(udot_m)


dict_e = {'rms_e_nm': rms_e_nm, 'rms_e_m': rms_e_m, 'rms_edot_nm': rms_edot_nm, 'rms_edot_m': rms_edot_m}
dict_u = {'rms_u_nm': rms_u_nm, 'rms_u_m': rms_u_m, 'rms_udot_nm': rms_udot_nm, 'rms_udot_m': rms_udot_m}

# ------------------- rms --------------------
cluster_e = np.concatenate((rms_e_nm, rms_e_m), axis=1).reshape(-1, 1)
cluster_edot = np.concatenate((rms_edot_nm, rms_edot_m), axis=1).reshape(-1, 1)

cluster_u = np.concatenate((rms_u_nm, rms_u_m), axis=1).reshape(-1, 1)
cluster_udot = np.concatenate((rms_udot_nm, rms_udot_m), axis=1).reshape(-1, 1)

# ------------------- non-rms ----------------
# cluster_e = np.concatenate((e_nm, e_m), axis=1).reshape(-1, 1)
# cluster_edot = np.concatenate((edot_nm, edot_m), axis=1).reshape(-1, 1)

# cluster_u = np.concatenate((u_nm, u_m), axis=1).reshape(-1, 1)
# cluster_udot = np.concatenate((udot_nm, udot_m), axis=1).reshape(-1, 1)

# ------------------- K-means -------------------
combined_data = np.concatenate((cluster_e, cluster_edot, cluster_u), axis=1)

n_clusters = 2

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(combined_data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
inertia = kmeans.inertia_
score = kmeans.score(combined_data)

centroids_diff = np.linalg.norm(centroids[0] - centroids[1])

print("Scalar difference between centroids:", centroids_diff)

print("Lables:", labels)

print("Inertia:", inertia)

print("Score:", score)

# ------------------- 3D K-means Plot -------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter1 = ax.scatter(combined_data[labels == 0, 0], combined_data[labels == 0, 1], combined_data[labels == 0, 2], 
                     c='red', alpha=0.5, label='Cluster 1')
scatter2 = ax.scatter(combined_data[labels == 1, 0], combined_data[labels == 1, 1], combined_data[labels == 1, 2], 
                     c='blue', alpha=0.5, label='Cluster 2')

centroid1 = ax.scatter(centroids[0, 0], centroids[0, 1], centroids[0, 2], marker='o', s=200, c='red', label='Centroid 1')
centroid2 = ax.scatter(centroids[1, 0], centroids[1, 1], centroids[1, 2], marker='o', s=200, c='blue', label='Centroid 2')

# Show legend
plt.legend(handles=[scatter1, scatter2, centroid1, centroid2])

ax.set_xlabel('e')
ax.set_ylabel('e_dot')
ax.set_zlabel('u')
ax.set_title('K-means Clustering')
plt.legend()
plt.show()