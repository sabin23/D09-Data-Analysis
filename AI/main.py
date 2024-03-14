import numpy as np

import matplotlib.pyplot as plt

from scipy.fftpack import fft

import scipy as sp

import scipy.io as sio


mat = sio.loadmat("C:/Users/jelme/Downloads/ae2224I_measurement_data_2024/ae2224I_measurement_data_subj1_C1.mat")
hpe = mat['Hpe_FC']

hper = np.real(hpe)
hpec = np.imag(hpe)

mag = np.sqrt(np.square(hper)+np.square(hpec))

freq = mat['w_FC']

print(freq)

plt.plot(freq, mag)
plt.xlabel('frequency')
plt.ylabel('magnitude')
plt.title('bode plot')
plt.grid(True)
plt.show()