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
# To do: find variance of each pilot for each run, then find average variance for each run and plot  
#Compile control and error data for NM case:
u = mat[1][1].u
u1 = np.concatenate(u)
t = mat[1][1].t

def sort_into_test_runs_NM(choice):
    u_1 = []
    u_2 = []
    u_3 = []
    u_4 = []
    u_5 = []
    for i in range(1,7):
        for j in range(1,4):
            if choice == 'e':
                matrix = mat[i-1][j-1].e
            if choice == 'u':
                matrix = mat[i-1][j-1].u
            u1 = []   #Creating empty sublists here which lets me put all values per run per pilot
            u2 = []
            u3 = []
            u4 = []
            u5 = []
            for value in matrix:
                u1.append(value[0])
                u2.append(value[1])
                u3.append(value[2])
                u4.append(value[3])
                u5.append(value[4])
            u_1.append(np.var(u1)) #These append the variances of the sublists containing signal per run per pilot
            u_2.append(np.var(u2))
            u_3.append(np.var(u3))
            u_4.append(np.var(u4))
            u_5.append(np.var(u5))
    
    runs = [u_1, u_2, u_3, u_4, u_5]
    return runs

def sort_into_test_runs_M(choice):
    u_1 = []
    u_2 = []
    u_3 = []
    u_4 = []
    u_5 = []
    for i in range(1,7):
        for j in range(4,7):
            if choice == 'e':
                matrix = mat[i-1][j-1].e
            if choice == 'u':
                matrix = mat[i-1][j-1].u
            u1 = []   #Creating empty sublists here which lets me put all values per run per pilot
            u2 = []
            u3 = []
            u4 = []
            u5 = []
            for value in matrix:
                u1.append(value[0])
                u2.append(value[1])
                u3.append(value[2])
                u4.append(value[3])
                u5.append(value[4])
            u_1.append(np.var(u1)) #These append the variances of the sublists containing signal per run per pilot
            u_2.append(np.var(u2))
            u_3.append(np.var(u3))
            u_4.append(np.var(u4))
            u_5.append(np.var(u5))
    
    runs = [u_1, u_2, u_3, u_4, u_5]
    return runs

def average_variance_per_run(runs):
    average_variances = []
    for run in runs:
        average_variances.append(np.mean(run))
    return average_variances

e_runs_NM = sort_into_test_runs_NM(choice = 'e')
e_average_variances_NM = average_variance_per_run(e_runs_NM)

e_runs_M = sort_into_test_runs_M(choice = 'e')
e_average_variances_M = average_variance_per_run(e_runs_M)

u_runs_NM = sort_into_test_runs_NM(choice = 'u')
u_average_variances_NM = average_variance_per_run(u_runs_NM)

u_runs_M = sort_into_test_runs_M(choice = 'u')
u_average_variances_M = average_variance_per_run(u_runs_M)

#Checking data
print(e_average_variances_NM)
print(e_average_variances_M)
print(u_average_variances_NM)
print(u_average_variances_M)

#Plotting
run = [1, 2, 3, 4, 5]
#TODO: Do regression with stats instead of numpy maybe???
m, b, c, d = np.polyfit(np.array(run), np.array(e_average_variances_NM), 3)
#m*(np.array(run))**2 + b*np.array(run) + c
plt.subplot(1,2,1)
plt.xlabel("Number of runs (1-5)")
plt.ylabel("Average Error Signal Variance")
plt.grid()
plt.scatter(run, e_average_variances_NM, marker = "o")
plt.plot(np.array(run), m*(np.array(run))**3 + b*(np.array(run))**2 + c*(np.array(run)) + d)
plt.plot(run, e_average_variances_M, marker = 'x')
plt.subplot(1,2,2)
plt.xlabel("Number of runs (1-5)")
plt.ylabel("Average Control Signal Variance")
plt.grid()
plt.plot(run, u_average_variances_NM, marker = 'o')
plt.plot(run, u_average_variances_M, marker = 'x')

plt.show()



#Code graveyard down here
""" ***OLD METHOD***
def sort_into_test_runs_NM2():
    u_1 = []
    u_2 = []
    u_3 = []
    u_4 = []
    u_5 = []
    for i in range(1,7):
        for j in range(1,4):
            matrix = mat[i-1][j-1].e
            for value in matrix:
                u_1.append(value[0])
                u_2.append(value[1])
                u_3.append(value[2])
                u_4.append(value[3])
                u_5.append(value[4])

    runs = [u_1, u_2, u_3, u_4, u_5]
    return runs

def sort_into_test_runs_M2():
    u_1 = []
    u_2 = []
    u_3 = []
    u_4 = []
    u_5 = []
    for i in range(1,7):
        for j in range(4,7):
            matrix = mat[i-1][j-1].e
            for value in matrix:
                u_1.append(value[0])
                u_2.append(value[1])
                u_3.append(value[2])
                u_4.append(value[3])
                u_5.append(value[4])

    runs = [u_1, u_2, u_3, u_4, u_5]
    return runs

runs_NM2 = sort_into_test_runs_NM2()
runs_M2 = sort_into_test_runs_M2()

def turn_into_list(runs):
    variances = []
    for run in runs:
        variances.append(np.var(run))

    return variances

variances_u_NM = turn_into_list(runs_NM2)
variances_u_M = turn_into_list(runs_M2)
    
print(variances_u_NM)
print(variances_u_M)

plt.plot(run, variances_u_NM)
plt.plot(run, variances_u_M)
plt.show()

"""
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

