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

# To do: find variance of each pilot for each run, then find average variance for each run and plot  
#Compile control and error data for NM case:

def sort_into_test_runs(choice, p1, p2, a, b):
    u_1 = [] 
    u_2 = []
    u_3 = []
    u_4 = []
    u_5 = []
    for i in range(p1,p2):
        for j in range(a,b):
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

#Sort data into P_NM, P_M, V_NM, V_M, A_NM, A_M in order to plot it as box and whisker plots.
P_NM = []
for i in range(2,8):
    P_NM.append(np.mean(sort_into_test_runs('e', i-1, i, 1, 2)))

P_M = []
for i in range(2,8):
    P_M.append(np.mean(sort_into_test_runs('e', i-1, i, 4, 5)))

V_NM = []
for i in range(2,8):
    V_NM.append(np.mean(sort_into_test_runs('e', i-1, i, 2, 3)))

V_M = []
for i in range(2,8):
    V_M.append(np.mean(sort_into_test_runs('e', i-1, i, 5, 6)))

A_NM = []
for i in range(2,8):
    A_NM.append(np.mean(sort_into_test_runs('e', i-1, i, 3, 4)))

A_M = []
for i in range(2,8):
    A_M.append(np.mean(sort_into_test_runs('e', i-1, i, 6, 7)))


##########


P_NMu = []
for i in range(2,8):
    P_NMu.append(np.mean(sort_into_test_runs('u', i-1, i, 1, 2)))

P_Mu = []
for i in range(2,8):
    P_Mu.append(np.mean(sort_into_test_runs('u', i-1, i, 4, 5)))

V_NMu = []
for i in range(2,8):
    V_NMu.append(np.mean(sort_into_test_runs('u', i-1, i, 2, 3)))

V_Mu = []
for i in range(2,8):
    V_Mu.append(np.mean(sort_into_test_runs('u', i-1, i, 5, 6)))

A_NMu = []
for i in range(2,8):
    A_NMu.append(np.mean(sort_into_test_runs('u', i-1, i, 3, 4)))

A_Mu = []
for i in range(2,8):
    A_Mu.append(np.mean(sort_into_test_runs('u', i-1, i, 6, 7)))
    

e_box = [np.array(P_NM),np.array(P_M),np.array(V_NM),np.array(V_M),np.array(A_NM),np.array(A_M)]
print(e_box)

u_box = [np.array(P_NMu),np.array(P_Mu),np.array(V_NMu),np.array(V_Mu),np.array(A_NMu),np.array(A_Mu)]
print(u_box)




fig, ax = plt.subplots()

# Create boxplots side by side

# ax.boxplot(e_box)
# ax.set_xticklabels(["$P_{NM}$", "$P_{M}$", "$V_{NM}$", "$V_{M}$", "$A_{NM}$", "$A_{M}$"])

# # Add title and labels
# plt.title('Boxplot')
# plt.xlabel('Conditions')
# plt.ylabel('Average Error Variance')


ax.boxplot(u_box)
ax.set_xticklabels(["$P_{NM}$", "$P_{M}$", "$V_{NM}$", "$V_{M}$", "$A_{NM}$", "$A_{M}$"])

# Add title and labels
plt.title('Boxplot')
plt.xlabel('Conditions')
plt.ylabel('Average Control Variance')



# Show the plot
plt.show()




# def average_variance_per_run(runs):
#     average_variances = []
#     for run in runs:
#         average_variances.append(np.mean(run))
#     return average_variances

# e_runs_NM = sort_into_test_runs('e',1,2)
# e_average_variances_NM = average_variance_per_run(e_runs_NM)

# e_runs_M = sort_into_test_runs('e',4,5)
# e_average_variances_M = average_variance_per_run(e_runs_M)

# u_runs_NM = sort_into_test_runs('u',1,2)
# u_average_variances_NM = average_variance_per_run(u_runs_NM)

# u_runs_M = sort_into_test_runs('u',4,5)
# u_average_variances_M = average_variance_per_run(u_runs_M)

# #Checking data
# print(e_average_variances_NM)
# print(e_average_variances_M)
# print(u_average_variances_NM)
# print(u_average_variances_M)



# # To do: find variance of each pilot for each run, then find average variance for each run and plot

# # Compile control and error data for NM case:
# u_NM = sort_into_test_runs('u',1,4)
# e_NM = sort_into_test_runs('e',1,4)

# # Compile control and error data for M case:

# u_M = sort_into_test_runs('u',4,7)
# e_M = sort_into_test_runs('e',4,7)

# # Calculate variances for each pilot and run in NM case:
# variances_u_NM = []
# variances_e_NM = []
# for i in range(5):
#     variances_u_NM.append([np.var(run) for run in u_NM[i]])
#     variances_e_NM.append([np.var(run) for run in e_NM[i]])

# # Calculate variances for each pilot and run in M case:
# variances_u_M = []
# variances_e_M = []
# for i in range(5):
#     variances_u_M.append([np.var(run) for run in u_M[i]])
#     variances_e_M.append([np.var(run) for run in e_M[i]])

# # Calculate average variances for each run in NM case:
# average_variances_u_NM = average_variance_per_run(variances_u_NM)
# average_variances_e_NM = average_variance_per_run(variances_e_NM)

# # Calculate average variances for each run in M case:
# average_variances_u_M = average_variance_per_run(variances_u_M)
# average_variances_e_M = average_variance_per_run(variances_e_M)

# print(len(average_variances_e_NM))
# print(len(average_variances_e_M))
# print(len(average_variances_u_NM))
# print(len(average_variances_u_M))

# # Plotting
# run = [1, 2, 3, 4, 5]
# plt.subplot(1, 2, 1)
# plt.ylim(0,27)
# plt.xlabel("Number of runs (1-5)")
# plt.ylabel("Average Error Variance")
# plt.grid()
# plt.plot(run, e_average_variances_NM, marker="o", label="NM")
# plt.plot(run, e_average_variances_M, marker="o", label="M")

# plt.legend()
# plt.subplot(1, 2, 2)
# plt.ylim(0,27)
# plt.xlabel("Number of runs (1-5)")
# plt.ylabel("Average Control Variance")
# plt.grid()
# plt.plot(run, u_average_variances_NM, marker="o", label="NM")
# plt.plot(run, u_average_variances_M, marker="o", label="M")
# plt.legend()
# plt.savefig('Variance.png')
# plt.show()

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

# def sort_into_test_runs_M(choice):
#     u_1 = []
#     u_2 = []
#     u_3 = []
#     u_4 = []
#     u_5 = []
#     for i in range(1,7):
#         for j in range(4,5):
#             if choice == 'e':
#                 matrix = mat[i-1][j-1].e
#             if choice == 'u':
#                 matrix = mat[i-1][j-1].u
#             u1 = []   #Creating empty sublists here which lets me put all values per run per pilot
#             u2 = []
#             u3 = []
#             u4 = []
#             u5 = []
#             for value in matrix:
#                 u1.append(value[0])
#                 u2.append(value[1])
#                 u3.append(value[2])
#                 u4.append(value[3])
#                 u5.append(value[4])
#             u_1.append(np.var(u1)) #These append the variances of the sublists containing signal per run per pilot
#             u_2.append(np.var(u2))
#             u_3.append(np.var(u3))
#             u_4.append(np.var(u4))
#             u_5.append(np.var(u5))
    
#     runs = [u_1, u_2, u_3, u_4, u_5]
#     return runs

# u = mat[1][1].u
# u1 = np.concatenate(u)
# t = mat[1][1].t

"""
ft = mat[2][2].ft
fd = mat[2][2].fd
u = mat[2][2].u
t = mat[2][2].t
e = mat[2][2].e
x = mat[2][2].x

print(u)

""" 

# #Plotting
# run = [1, 2, 3, 4, 5]
# #TODO: Do regression with stats instead of numpy maybe???
# m, b, c, d = np.polyfit(np.array(run), np.array(e_average_variances_NM), 3)
# #m*(np.array(run))**2 + b*np.array(run) + c
# plt.subplot(1,2,1)
# plt.xlabel("Number of runs (1-5)")
# plt.ylabel("Average Error Signal Variance")
# plt.grid()
# plt.scatter(run, e_average_variances_NM, marker = "o")
# plt.plot(np.array(run), m*(np.array(run))**3 + b*(np.array(run))**2 + c*(np.array(run)) + d)
# plt.plot(run, e_average_variances_M, marker = 'x')
# plt.subplot(1,2,2)
# plt.xlabel("Number of runs (1-5)")
# plt.ylabel("Average Control Signal Variance")
# plt.grid()
# plt.plot(run, u_average_variances_NM, marker = 'o')
# plt.plot(run, u_average_variances_M, marker = 'x')

# plt.show()