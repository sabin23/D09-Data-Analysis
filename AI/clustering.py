import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data

# nm -> no motion; m -> motion; (pilot, test)
data_nm = Data(3, 3)
data_m = Data(3, 4)

e_nm = data_nm.e
e_m = data_m.e

x_nm = data_nm.x
x_m = data_m.x

u_nm = data_nm.u
u_m = data_m.u

t = data_nm.t


def rms(f):
    return np.sqrt((np.mean(f**2, axis=1)))


rms_e_nm = rms(e_nm)
rms_e_m = rms(e_m)

rms_u_nm = rms(u_nm)
rms_u_m = rms(u_m)


t = t.reshape(-1, 1)


def dfdt(f, t):
    dfdt = np.zeros(f.shape)
    for i in range(0, len(f) - 1):
        if i > 0 and i < len(f) - 1:
            dfdt[i] = (f[i + 1] - f[i - 1]) / (t[i + 1] - t[i - 1])
    dfdt[len(f) - 1] = dfdt[len(f) - 2]
    dfdt[0] = dfdt[1]
    return dfdt


edot_nm = dfdt(e_nm, t)
edot_m = dfdt(e_m, t)

udot_nm = dfdt(u_nm, t)
udot_m = dfdt(u_m, t)


rms_edot_nm = rms(edot_nm)
rms_edot_m = rms(edot_m)

rms_udot_nm = rms(udot_nm)
rms_udot_m = rms(udot_m)


def mean(f):
    return np.mean(f, axis=1).reshape(-1, 1)


e_nm = mean(e_nm)
e_m = mean(e_m)

u_nm = mean(u_nm)
u_m = mean(u_m)

edot_nm = mean(edot_nm)
edot_m = mean(edot_m)

udot_nm = mean(udot_nm)
udot_m = mean(udot_m)


def rms_slices(f, n):
    slices = np.split(f, n)
    return np.array([np.sqrt(np.mean(slice**2)) for slice in slices])


for i in range(1, 14):
    if 8192 % (pow(2, i)) != 0:
        print("8192 is not divisible by 2^", i)
        continue
    slices_e_nm = rms_slices(rms_e_nm, pow(2, i)).reshape(-1, 1)
    slices_e_m = rms_slices(rms_e_m, pow(2, i)).reshape(-1, 1)
    slices_edot_nm = rms_slices(rms_edot_nm, pow(2, i)).reshape(-1, 1)
    slices_edot_m = rms_slices(rms_edot_m, pow(2, i)).reshape(-1, 1)
    slices_u_nm = rms_slices(rms_u_nm, pow(2, i)).reshape(-1, 1)
    slices_u_m = rms_slices(rms_u_m, pow(2, i)).reshape(-1, 1)
    slices_udot_nm = rms_slices(rms_udot_nm, pow(2, i)).reshape(-1, 1)
    slices_udot_m = rms_slices(rms_udot_m, pow(2, i)).reshape(-1, 1)

    dict_e = {
        "rms_e_nm": rms_e_nm,
        "rms_e_m": rms_e_m,
        "rms_edot_nm": rms_edot_nm,
        "rms_edot_m": rms_edot_m,
    }
    dict_u = {
        "rms_u_nm": rms_u_nm,
        "rms_u_m": rms_u_m,
        "rms_udot_nm": rms_udot_nm,
        "rms_udot_m": rms_udot_m,
    }

    # ------------------- 2 sets of data --------------------
    cluster_e = np.concatenate((slices_e_nm, slices_e_m), axis=0)
    cluster_edot = np.concatenate((slices_edot_nm, slices_edot_m), axis=0)

    cluster_u = np.concatenate((slices_u_nm, slices_u_m), axis=0)
    cluster_udot = np.concatenate((slices_udot_nm, slices_udot_m), axis=0)

    combined_data_with_labels = np.concatenate(
        (np.zeros((pow(2, i), 1)), np.ones((pow(2, i), 1))), axis=0
    )

    # ------------------- 1 set of data --------------------
    # cluster_e = slices_e_m.reshape(-1, 1)
    # cluster_edot = slices_edot_m.reshape(-1, 1)

    # cluster_u = slices_u_m.reshape(-1, 1)
    # cluster_udot = slices_udot_m.reshape(-1, 1)

    # combined_data_with_labels = np.concatenate((np.zeros((pow(2, i-1), 1)), np.ones((pow(2, i-1), 1))), axis=0)

    # ------------------- K-means -------------------
    combined_data = np.concatenate(
        (cluster_e, cluster_edot, cluster_u, cluster_udot), axis=1
    )

    n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(combined_data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_

    centroids_diff = np.linalg.norm(centroids[0] - centroids[1])

    print("Number of slices:", pow(2, i))

    print("Scalar difference between centroids:", centroids_diff)

    cluster_0_amount_of_nm = 0
    cluster_0_amount_of_m = 0
    cluster_1_amount_of_nm = 0
    cluster_1_amount_of_m = 0

    for number in combined_data_with_labels[labels == 0, 0]:
        if number == 0:
            cluster_0_amount_of_nm += 1
        elif number == 1:
            cluster_0_amount_of_m += 1

    for number in combined_data_with_labels[labels == 1, 0]:
        if number == 0:
            cluster_1_amount_of_nm += 1
        elif number == 1:
            cluster_1_amount_of_m += 1

    print(
        "Cluster 0 amount of no motion:",
        round(
            cluster_0_amount_of_nm
            / (cluster_0_amount_of_nm + cluster_0_amount_of_m)
            * 100
        ),
        "%",
    )
    print(
        "Cluster 0 amount of motion:",
        round(
            cluster_0_amount_of_m
            / (cluster_0_amount_of_nm + cluster_0_amount_of_m)
            * 100
        ),
        "%",
    )
    print(
        "Cluster 1 amount of no motion:",
        round(
            cluster_1_amount_of_nm
            / (cluster_1_amount_of_nm + cluster_1_amount_of_m)
            * 100
        ),
        "%",
    )
    print(
        "Cluster 1 amount of motion:",
        round(
            cluster_1_amount_of_m
            / (cluster_1_amount_of_nm + cluster_1_amount_of_m)
            * 100
        ),
        "%",
    )

    print("Inertia/number of slices:", inertia / pow(2, i))

    # ------------------- 3D K-means Plot -------------------

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    scatter1 = ax.scatter(
        combined_data[labels == 0, 0],
        combined_data[labels == 0, 1],
        combined_data[labels == 0, 2],
        c=combined_data[labels == 0, 3],
        alpha=0.5,
        label="Cluster 1",
        cmap=plt.hot(),
    )
    scatter2 = ax.scatter(
        combined_data[labels == 1, 0],
        combined_data[labels == 1, 1],
        combined_data[labels == 1, 2],
        c=combined_data[labels == 1, 3],
        alpha=0.5,
        label="Cluster 2",
        cmap=plt.cool(),
    )
    # scatter3 = ax.scatter(combined_data[labels == 2, 0], combined_data[labels == 2, 1], combined_data[labels == 2, 2],
    # c='green', alpha=0.5, label='Cluster 3')

    centroid1 = ax.scatter(
        centroids[0, 0],
        centroids[0, 1],
        centroids[0, 2],
        marker="o",
        s=200,
        c="red",
        label="Centroid 1",
    )
    centroid2 = ax.scatter(
        centroids[1, 0],
        centroids[1, 1],
        centroids[1, 2],
        marker="o",
        s=200,
        c="blue",
        label="Centroid 2",
    )
    # centroid3 = ax.scatter(centroids[2, 0], centroids[2, 1], centroids[2, 2], marker='o', s=200, c='green', label='Centroid 3')

    # Show legend
    plt.legend(handles=[scatter1, scatter2, centroid1, centroid2])
    # plt.legend(handles=[scatter1, scatter2, scatter3, centroid1, centroid2, centroid3])

    ax.set_xlabel("e")
    ax.set_ylabel("e_dot")
    ax.set_zlabel("u")
    ax.set_title("K-means Clustering")
    fig.colorbar(scatter1)
    fig.colorbar(scatter2)
    plt.legend()
    plt.show()
