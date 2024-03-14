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

"""
ft = mat[2][2].ft
fd = mat[2][2].fd
u = mat[2][2].u
t = mat[2][2].t
e = mat[2][2].e
x = mat[2][2].x

print(u)

"""

#Compile control and error data for NM case:
u = mat[1][1].u
u1 = np.concatenate(u)
t = mat[1][1].t

def sort_into_test_runs_NM():
    u_1 = []
    u_2 = []
    u_3 = []
    u_4 = []
    u_5 = []
    for i in range(1,7):
        for j in range(2,3):
            matrix = mat[i-1][j-1].e
            for value in matrix:
                u_1.append(value[0])
                u_2.append(value[1])
                u_3.append(value[2])
                u_4.append(value[3])
                u_5.append(value[4])

    runs = [u_1, u_2, u_3, u_4, u_5]
    return runs

def sort_into_test_runs_M():
    u_1 = []
    u_2 = []
    u_3 = []
    u_4 = []
    u_5 = []
    for i in range(1,7):
        for j in range(5,6):
            matrix = mat[i-1][j-1].e
            for value in matrix:
                u_1.append(value[0])
                u_2.append(value[1])
                u_3.append(value[2])
                u_4.append(value[3])
                u_5.append(value[4])

    runs = [u_1, u_2, u_3, u_4, u_5]
    return runs

u_runs_NM = sort_into_test_runs_NM()
u_runs_M = sort_into_test_runs_M()

def turn_into_list(runs):
    variances = []
    for run in runs:
        variances.append(np.var(run))

    return variances

variances_u_NM = turn_into_list(u_runs_NM)
variances_u_M = turn_into_list(u_runs_M)
    





run = [1, 2, 3, 4, 5]


            

plt.plot(run, variances_u_NM)
plt.plot(run, variances_u_M)
plt.show()

    
















#u_run1 = []
#u_run2 = []
#u_run3 = []
#u_run4 = []
#u_run5 = []

"""

for value in u:
    u_run1.append(value[0])
    u_run2.append(value[1])
    u_run3.append(value[2])
    u_run4.append(value[3])
    u_run5.append(value[4])

var_u1 = np.var(u_run1)
var_u2 = np.var(u_run2)
var_u3 = np.var(u_run3)
var_u4 = np.var(u_run4)
var_u5 = np.var(u_run5)

"""
#variances_u1 = [var_u1, var_u2, var_u3, var_u4, var_u5]
#print(len(u_avg))

#plt.plot(np.transpose(t), u_run1)
#plt.plot(np.transpose(t), u_run2)
#plt.plot(np.transpose(t), u_run3)
#plt.plot(np.transpose(t), u_run4)
#plt.plot(np.transpose(t), u_run5)

#print(u)


#print(u1)
#print(t)

