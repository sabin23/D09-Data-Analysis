import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from get_data import Data

# Load the data
mat = []
for i in range(1, 7):
    line = []
    for j in range(1, 7):
        line.append(Data(i, j))
    mat.append(line)

ft = mat[2][2].ft
fd = mat[2][2].fd
u = mat[2][2].u
t = mat[2][2].t
e = mat[2][2].e
x = mat[2][2].x

print(u)