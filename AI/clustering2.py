import torch
import numpy as np

from get_data import Data
from sklearn.cluster import AgglomerativeClustering

def dfdt (f, t):
    dfdt = np.zeros(f.shape)
    for i in range(0, len(f)-1):
        if i > 0 and i < len(f)-1:
            dfdt[i] = (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
    dfdt[len(f)-1] = dfdt[len(f)-2]
    dfdt[0] = dfdt[1]
    return dfdt

data = Data(1, 1)

u = Data(1, 1).u
e = Data(1, 1).e

t = Data(1, 1).t[0]
x = Data(1, 1).x

x_t = x.transpose(1, 0)
dxdt = np.zeros(x_t.shape)
for i in range(0,len(x_t)):
    dxdt[i] = dfdt(x_t[i], t)
dxdt = dxdt.transpose(1, 0)
print(dxdt)

import matplotlib.pyplot as plt

# Combine dxdt and e into a single feature matrix
features = np.column_stack((dxdt, e))

# Create an instance of AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=2)

# Fit the clustering model to the data
labels = clustering.fit_predict(features)

# Plot the result
plt.scatter(dxdt[:, 0], dxdt[:, 1], c=labels)
plt.xlabel('dxdt')
plt.ylabel('e')
plt.title('Agglomerative Hierarchical Clustering')
plt.show()