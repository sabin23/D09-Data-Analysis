import sys

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from get_data import Data


data = Data(1, 1)

u = data.u
e = data.e
t = data.t


def get_rms(data):
    return np.sqrt(np.mean(data**2))


def get_dot(data, t):
    return np.diff(data, axis=0) / np.diff(t, axis=0)


e_mean = np.mean(u, axis=1)
e_rms = np.sqrt(np.mean(e, axis=1) ** 2)

print(e_rms, e_rms.shape)


def rms_slices(data, slices):
    slices = np.split(data, slices)
    return np.array([get_rms(_slice) for _slice in slices])


variancesMax = []
variancesMin = []
number_of_samples = []

variance_e = np.var(e_mean)

for i in range(1, 8):
    if 8192 % (pow(2, i)) != 0:
        print(f"Skipping splitting by {pow(2,i)} because: ")
        print(8192 % (pow(2, i)))
        continue
    slices = np.split(e_mean, pow(2, i))

    rms_slices = np.array([get_rms(_slice) for _slice in slices])
    print(f"RMS for {pow(2,i)}: {rms_slices}")
    print("RMS slices shape: ", rms_slices.shape)

    continue
    number_of_samples.append(len(slices))
    max_variance = 0
    min_variance = 100000000
    slices_in_bounds = 0
    for _slice in slices:
        variance = np.var(_slice)
        variance_diff = variance / variance_e
        if variance_diff > max_variance:
            max_variance = variance_diff
        if variance_diff < min_variance:
            min_variance = variance_diff
        if variance > 1.15 * variance_e:
            continue
        # print("Variance inside bounds")
        # print(len(slices))
        slices_in_bounds += 1
    print(f"Number of slices in bounds for {pow(2,i)}: {slices_in_bounds}")
    variancesMax.append(max_variance)
    variancesMin.append(min_variance)

variancesMax = np.array(variancesMax)
variancesMin = np.array(variancesMin)
number_of_samples = np.array(number_of_samples)

# Create the plot
plt.figure(figsize=(10, 5))  # Set the size of the plot

plt.plot(number_of_samples, variancesMax, label="Max Variances", color="blue")
plt.plot(number_of_samples, variancesMin, label="Min Variances", color="red")
plt.plot(
    number_of_samples, variancesMax - variancesMin, label="Difference", color="green"
)

# Customize the plot
plt.title("Comparison of Variances")  # Title of the plot
plt.xlabel("Number of Samples")  # X-axis label
plt.ylabel("Variance")  # Y-axis label
plt.legend()  # Add a legend to distinguish the lines

# Show the plot
plt.show()

print(e_mean, e_mean.shape)
