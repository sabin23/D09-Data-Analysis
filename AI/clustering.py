import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data

# nm -> no motion; m -> motion; (pilot, test)
data_nm = Data(4, 3)
data_m = Data(4, 4)

e_nm = data_nm.e
e_m = data_m.e

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

def rms_slices(f, n):
    slices = np.split(f, n)
    return np.array([np.sqrt(np.mean(slice**2)) for slice in slices])


inertia_list = []
number_of_slices_list = []
inertia_per_slice_list = []
percentage_diff_list = []


def standardize(data):
    # Compute the mean and std of the dataset
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)

    # Standardize the data
    standardized_data = (data - mean) / std
    return standardized_data


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

    # ------------------- 2 sets of data --------------------
    cluster_e = np.concatenate((slices_e_nm, slices_e_m), axis=0)
    cluster_edot = np.concatenate((slices_edot_nm, slices_edot_m), axis=0)
    cluster_e = np.concatenate((slices_e_nm, slices_e_m), axis=0)
    cluster_edot = np.concatenate((slices_edot_nm, slices_edot_m), axis=0)

    cluster_u = np.concatenate((slices_u_nm, slices_u_m), axis=0)
    cluster_udot = np.concatenate((slices_udot_nm, slices_udot_m), axis=0)
    cluster_u = np.concatenate((slices_u_nm, slices_u_m), axis=0)
    cluster_udot = np.concatenate((slices_udot_nm, slices_udot_m), axis=0)

    combined_data_with_labels = np.concatenate(
        (np.zeros((pow(2, i), 1)), np.ones((pow(2, i), 1))), axis=0
    )

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

    print("Scalar distance between centroids:", centroids_diff)

    cluster_1_amount_of_nm = 0
    cluster_2_amount_of_nm = 0
    cluster_1_amount_of_m = 0
    cluster_2_amount_of_m = 0

    for number in combined_data_with_labels[labels == 0, 0]:
        if number == 0:
            cluster_1_amount_of_nm += 1
        elif number == 1:
            cluster_1_amount_of_m += 1

    for number in combined_data_with_labels[labels == 1, 0]:
        if number == 0:
            cluster_2_amount_of_nm += 1
        elif number == 1:
            cluster_2_amount_of_m += 1

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
    print(
        "Cluster 2 amount of no motion:",
        round(
            cluster_2_amount_of_nm
            / (cluster_2_amount_of_nm + cluster_2_amount_of_m)
            * 100
        ),
        "%",
    )
    print(
        "Cluster 2 amount of motion:",
        round(
            cluster_2_amount_of_m
            / (cluster_2_amount_of_nm + cluster_2_amount_of_m)
            * 100
        ),
        "%",
    )

    percentage_diff_cluster_1 = (
        1
        - abs(
            cluster_1_amount_of_nm / (cluster_1_amount_of_nm + cluster_1_amount_of_m)
            - cluster_1_amount_of_m / (cluster_1_amount_of_nm + cluster_1_amount_of_m)
        )
    ) * 100

    percentage_diff_cluster_2 = (
        1
        - abs(
            cluster_2_amount_of_nm / (cluster_2_amount_of_nm + cluster_2_amount_of_m)
            - cluster_2_amount_of_m / (cluster_2_amount_of_nm + cluster_2_amount_of_m)
        )
    ) * 100

    percentage_diff_list.append(
        (percentage_diff_cluster_1 + percentage_diff_cluster_2) / 2
    )

    inertia_list.append(inertia)

    number_of_slices_list.append(pow(2, i))

    inertia_per_slice_list.append(inertia / pow(2, i))

    print("Inertia per Slice:", inertia / pow(2, i))
    print("-----------------------------------")
    

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
        cmap=plt.winter(),
    )
    scatter2 = ax.scatter(
        combined_data[labels == 1, 0],
        combined_data[labels == 1, 1],
        combined_data[labels == 1, 2],
        c=combined_data[labels == 1, 3],
        alpha=0.5,
        label="Cluster 2",
        cmap=plt.hot(),
    )

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

    # Show legend
    plt.legend(handles=[scatter1, scatter2, centroid1, centroid2])

    ax.set_xlabel("e")
    ax.set_ylabel("e_dot")
    ax.set_zlabel("u")
    ax.set_title("K-means Clustering")
    colorbar_1 = fig.colorbar(scatter1)
    colorbar_2 = fig.colorbar(scatter2)
    colorbar_1.set_label("u_dot cluster 1")
    colorbar_2.set_label("u_dot cluster 2")
    plt.legend()
    plt.show()

# ------------------- Inertia per Slice and Percentage Difference vs Number of Slices -------------------

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# # Inertia per Slice vs Number of Slices
# ax1.scatter(number_of_slices_list, inertia_per_slice_list)
# ax1.set_xscale("log")
# ax1.set_xlabel("Number of Slices")
# ax1.set_ylabel("Inertia per Slice")
# ax1.set_title("Inertia per Slice vs Number of Slices")

# # Percentage Difference vs Number of Slices
# ax2.scatter(number_of_slices_list, percentage_diff_list)
# ax2.set_xscale("log")
# ax2.set_xlabel("Number of Slices")
# ax2.set_ylabel("Percentage Difference")
# ax2.set_title("Percentage Difference vs Number of Slices")

# plt.tight_layout()
# plt.show()