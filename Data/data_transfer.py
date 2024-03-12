import scipy.io as sio
from os.path import dirname, join as pjoin

mat = sio.loadmat('Data\ae2224I_measurement_data_subj1_C1.mat')

lines = mat.readlines()

for line in lines:
    print(line)